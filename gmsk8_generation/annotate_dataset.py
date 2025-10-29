import anthropic
import os
import json
import argparse
from datasets import load_from_disk

# Default configuration variables (overridden by command-line params)
DEFAULT_NUM_EXAMPLES = 10
DEFAULT_SPLIT = "test"
DATASET_PATH = "resources/gmsk8_dataset"

SYSTEM_PROMPT = """
# GSM8K Math Problem Annotation Instructions

You are annotating grade school math problems to extract fundamental mathematical and reasoning properties. Your output must be a valid JSON object with exactly 16 fields.

## Input Format

You will receive:
- **question**: The problem text
- **answer**: The solution with step-by-step calculations showing all arithmetic operations

## Your Task

Analyze the solution's mathematical operations and reasoning structure, then output a JSON annotation with 16 fields divided into two tiers.

---

# TIER 1: DIRECT PROBING FEATURES (11 fields)

These features test what the model can directly detect and compute.

## 1. ops (array of strings)

**Description:** List all arithmetic operations in execution order.

**Valid values:** `"add"`, `"sub"`, `"mult"`, `"div"`

**Rules:**
- Extract operations from the solution in the exact order they are calculated
- Include only arithmetic operations (not intermediate conversions)
- For percentage calculations like "200 * 40% = 80", this is TWO operations: `["mult", "div"]` (200*40=8000, then 8000/100=80)

**Example:**
```
Solution: 180 * 1 = 180, then 365 - 180 = 185, then 185 * 2 = 370, then 180 + 370 = 550, then 550 / 110 = 5
ops: ["mult", "sub", "mult", "add", "div"]
```

---

## 2. steps (integer)

**Description:** Total number of calculation steps.

**Rules:**
- Count each arithmetic operation as one step
- Must equal `length(ops)`

**Example:**
```
ops: ["mult", "sub", "mult", "add", "div"]
steps: 5
```

---

## 3. can_parallelize (boolean)

**Description:** Can ANY operations be computed independently (in parallel)?

**Rules:**
- `true`: At least 2 operations can be done simultaneously (don't depend on each other)
- `false`: All operations form a strict chain (each needs the previous result)

**How to test:**
- Look at the first few operations
- Ask: "Can operation 2 be computed without waiting for operation 1's result?"
- If ANY pair of operations are independent then true
- If EVERY operation needs all previous operations then false

**Examples:**

```
# can_parallelize: true
1. 3 + 4 = 7        (independent)
2. 2 + 5 = 7        (independent) - Can compute at same time as step 1!
3. 7 + 7 = 14       (needs both)
```

```
# can_parallelize: false (pure chain)
1. 10 + 5 = 15
2. 15 * 2 = 30      (needs step 1's result)
3. 30 - 10 = 20     (needs step 2's result)
```

---

## 4. max_num (integer)

**Description:** The largest number appearing anywhere in the problem.

**Rules:**
- Scan ALL numbers: in problem text, operands, and intermediate results
- Include the final answer
- Return the maximum value

**Example:**
```
Problem mentions: 200, 40, 2, 20
Calculations produce: 8000, 80, 40, 100, 140, 160
max_num: 8000
```

---

## 5. max_operand_digits (integer)

**Description:** Maximum number of digits in any operand used in calculations.

**Rules:**
- For each operation, count digits in both operands
- For whole numbers: 365 has 3 digits, 5 has 1 digit
- For decimals: count only significant digits (0.5 has 1 digit, 1.5 has 1 digit, 0.25 has 2 digits)
- Return the maximum across all operands

**Example:**
```
180 * 1 = 180       (operands: 3 digits, 1 digit)
8000 / 100 = 80     (operands: 4 digits, 3 digits)
max_operand_digits: 4
```

---

## 6. has_decimals (boolean)

**Description:** Do any non-integer numbers appear in the problem?

**Check for:**
- Decimal points: 0.5, 1.5, 2.25
- Fractions: 1/2, 3/4
- Percentages that become decimals: 40% becomes 0.4 or requires division

**Rules:**
- `true`: Any decimal, fraction, or percentage appears
- `false`: All numbers are whole integers throughout

**Example:**
```
30 * 0.5 = 15       has_decimals: true
200 * 40% = 80      has_decimals: true (involves 0.4 or /100)
30 * 2 = 60         has_decimals: false
```

---

## 7. has_carrying (boolean)

**Description:** Does ANY addition operation require carrying?

**How to check:**
1. For each addition, write it vertically
2. Add each column from right to left
3. If any column sums to >=10, carrying is needed

**Examples:**

```
  180
+ 370
-----
Ones: 0+0 = 0 (no carry)
Tens: 8+7 = 15 (>=10, CARRYING REQUIRED!)
has_carrying: true
```

```
   23
+  34
-----
Ones: 3+4 = 7 (no carry)
Tens: 2+3 = 5 (no carry)
has_carrying: false
```

**Rules:**
- Check ALL additions in the problem
- If ANY addition requires carrying then true
- If no additions exist then false

---

## 8. has_borrowing (boolean)

**Description:** Does ANY subtraction operation require borrowing?

**How to check:**
1. For each subtraction, write it vertically
2. Check each column from right to left
3. If any top digit is less than bottom digit, borrowing is needed

**Examples:**

```
  365
- 180
-----
Ones: 5 >= 0 (ok)
Tens: 6 < 8 (BORROWING REQUIRED!)
has_borrowing: true
```

```
   89
-  23
-----
Ones: 9 >= 3 (ok)
Tens: 8 >= 2 (ok)
has_borrowing: false
```

**Rules:**
- Check ALL subtractions in the problem
- If ANY subtraction requires borrowing then true
- If no subtractions exist then false

---

## 9. div_exact (boolean or null)

**Description:** For division operations, is the result a whole number?

**Values:**
- `true`: All divisions result in whole numbers (no remainder)
- `false`: At least one division has a remainder
- `null`: No division operations in the problem

**Examples:**
```
550 / 110 = 5           div_exact: true (no remainder)
100 / 3 = 33.33...      div_exact: false (has remainder)
(no division present)   div_exact: null
```

**Rules:**
- If problem has multiple divisions, ALL must be exact for `true`
- If ANY division has a remainder then false

---

## 10. has_rate_or_ratio (boolean)

**Description:** Does the problem involve rates, ratios, or percentages?

**Look for:**
- Rates: "2 GB/minute", "60 mph", "5 dollars per hour", "cups per day"
- Percentages: "40%", "25 percent"
- Ratios: "3 to 1", "for every 2 apples"
- Keywords: "per", "each", "%", "every"

**Rules:**
- `true`: Any rate, ratio, or percentage is present
- `false`: Only absolute quantities (no rates)

**Examples:**
```
"downloads 2 GB/minute"           has_rate_or_ratio: true
"drives at 60 mph"                has_rate_or_ratio: true
"1 cup per day for 180 days"     has_rate_or_ratio: true
"John has 5 apples and Mary has 3" has_rate_or_ratio: false
```

---

## 11. requires_unit_conversion (boolean)

**Description:** Does the problem require converting between different units, using implicit quantity knowledge, or translating between different forms of measurement?

**Common conversions in GSM8K:**
- **Implicit quantities**: "year" = 365 days, "dozen" = 12, "hour" = 60 minutes, "day" = 24 hours
- **Percentage conversions**: 40% to 0.4 or /100
- **Fraction conversions**: "half" to 0.5, "quarter" to 0.25, "third" to 0.33
- **Rate conversions**: "per day" over multiple days, "per minute" over hours
- **Unit conversions**: feet to inches, dollars to cents, kg to pounds

**Rules:**
- `true`: Problem uses a quantity/unit that must be converted to a number (not explicitly stated)
- `false`: All quantities are explicit numbers, no conversion or implicit knowledge needed

**Examples:**

```
# REQUIRES CONVERSION (true):
"in the first year" must know year = 365 days           requires_unit_conversion: true
"half an hour" must convert "half" to 0.5               requires_unit_conversion: true  
"40% of the file" must convert to 0.4 or /100           requires_unit_conversion: true
"a dozen eggs" must know dozen = 12                     requires_unit_conversion: true
"per year" must know year = 365 days                    requires_unit_conversion: true

# NO CONVERSION NEEDED (false):
"150% is 1.5" problem states the conversion directly    requires_unit_conversion: false
"2 GB per minute for 40 minutes" 2*40 (no conversion)   requires_unit_conversion: false
"John has 5 apples, Mary has 3" all explicit numbers    requires_unit_conversion: false
"365 days" explicit number given                        requires_unit_conversion: false
```

**Key distinction:**
- **Requires conversion**: Implicit quantities that must be looked up/known ("year", "half", "dozen", "40%")
- **No conversion**: Explicit numbers already stated ("365 days", "0.5 hours", "12 items", "1.5")

**Test:** If you removed all your math knowledge and only knew basic arithmetic, would you still know what number to use? If no then requires conversion.

---

# TIER 2: STRUCTURAL ANALYSIS FEATURES (5 fields)

These features describe the reasoning structure and are used for correlation analysis.

## 12. requires_planning (boolean)

**Description:** Must you calculate one step before you can calculate another?

**Rules:**
- `true`: Operations have dependencies (at least some steps need previous results)
- `false`: All operations are completely independent

**Heuristic:**
- If `can_parallelize == false` AND `steps >= 3` then usually `true`
- If all steps are independent simple calculations then `false`

**Example:**
```
# requires_planning: true
Step 2 needs Step 1's result, Step 3 needs Step 2's result
Must plan the execution order!

# requires_planning: false  
All steps: 2+3, 4*5, 10-1 (all independent, no planning needed)
```

---

## 13. num_contexts (integer)

**Description:** Number of distinct scenarios, time periods, or entities being tracked separately.

**What counts as a context:**
- Different time periods: "first 180 days" vs "remaining days"
- Different people/entities: "John's apples" vs "Mary's apples"
- Different scenarios: "going to work" vs "coming back home"
- Different phases: "before restart" vs "after restart"

**Rules:**
- Count distinct contexts that are calculated separately
- Minimum is 1 (single context)
- Don't count the final aggregation as a separate context

**Examples:**
```
"feed 1 cup for first 180 days, then 2 cups for rest of year"
num_contexts: 2 (first period, second period)

"John has 3 apples and buys 4 more. Mary has 2 apples and buys 5 more."
num_contexts: 2 (John, Mary)

"drives for 3 hours, then turns around, spends 2 hours in traffic, then 0.5 hours at 30mph, then rest at 80mph"
num_contexts: 4 (outbound, traffic period, slow driving, fast driving)
```

---

## 14. intermediate_results_used (integer)

**Description:** How many intermediate calculation results are reused in later steps?

**How to count:**
1. Track each calculation's result
2. Count how many of these results are used as inputs to subsequent calculations
3. Don't count the final answer (it's not "reused")

**Example:**
```
Step 1: 180 * 1 = 180
Step 2: 365 - 180 = 185
Step 3: 185 * 2 = 370
Step 4: 180 + 370 = 550    REUSES results from steps 1 and 3
Step 5: 550 / 110 = 5      REUSES result from step 4

Intermediate results reused: 180 (used in step 4), 370 (used in step 4), 550 (used in step 5)
intermediate_results_used: 3
```

**Rules:**
- Only count results that are explicitly used later
- Don't count inputs that are constants from the problem
- Don't count the final answer

---

## 15. max_chain_depth (integer)

**Description:** Length of the longest dependency chain in the problem.

**How to calculate:**
1. Build a dependency graph: which steps depend on which
2. Find the longest path from any starting step to any ending step
3. Count the number of steps in that longest path

**Example:**
```
Step 1: independent (depth 1)
Step 2: independent (depth 1)
Step 3: needs step 2 (depth 2)
Step 4: needs step 3 (depth 3)
Step 5: needs steps 1 and 4 (depth max(1, 3)+1 = 4)

max_chain_depth: 4
```

**Simple heuristic:**
- If purely sequential (each needs previous): `max_chain_depth = steps`
- If some parallelism: trace the longest path

---

## 16. num_entities (integer)

**Description:** Number of distinct quantities or variables being tracked.

**What counts as an entity:**
- Different measured quantities: "cups in period 1", "cups in period 2", "total cups", "bags"
- Different objects: "John's apples", "Mary's apples", "combined apples"
- Different accumulations: "distance going", "distance returning", "net distance"

**Rules:**
- Count each distinct quantity that is calculated and represents something different
- Intermediate calculations of the same type count as one entity
- Final aggregations count as separate entities

**Example (puppy problem):**
```
Entity 1: cups for first period (180)
Entity 2: days remaining (185)
Entity 3: cups for second period (370)
Entity 4: total cups (550)
Entity 5: number of bags (5)
num_entities: 5
```

**Example (simple problem):**
```
"John has 3 apples and buys 4 more. How many does he have?"
Entity 1: John's initial apples (3)
Entity 2: John's total apples (7)
num_entities: 2
```

---

# OUTPUT FORMAT

**CRITICAL:** You must provide TWO things:

1. **The JSON annotation** (in a code block)
2. **Your reasoning** explaining why you chose each value (in plain text)

**Output Structure:**

```json
{
  "ops": ["mult", "sub", "mult", "add", "div"],
  "steps": 5,
  "can_parallelize": false,
  "max_num": 550,
  "max_operand_digits": 3,
  "has_decimals": false,
  "has_carrying": true,
  "has_borrowing": true,
  "div_exact": true,
  "has_rate_or_ratio": true,
  "requires_unit_conversion": true,
  "requires_planning": true,
  "num_contexts": 2,
  "intermediate_results_used": 4,
  "max_chain_depth": 4,
  "num_entities": 5
}
```

**Reasoning:**

ops: Reading through the solution: 180*1, 365-180, 185*2, 180+370, 550/110 = ["mult","sub","mult","add","div"]

steps: 5 operations total

can_parallelize: false - Step 2 needs result from step 1 (180), step 3 needs result from step 2 (185), forming a chain

max_num: 550 - largest number appearing (scanning: 180, 365, 185, 370, 550, 110, 5)

max_operand_digits: 3 - checking all operands: 180(3), 1(1), 365(3), 180(3), 185(3), 2(1), 180(3), 370(3), 550(3), 110(3). Max=3

has_decimals: false - all numbers are integers

has_carrying: true - checking 180+370: ones 0+0=0, tens 8+7=15 (>=10, CARRY!)

has_borrowing: true - checking 365-180: ones 5-0=5 (ok), tens 6<8 (BORROW!)

div_exact: true - 550/110=5 exactly, no remainder

has_rate_or_ratio: true - "1 cup per day", "2 cups per day" are rates

requires_unit_conversion: true - "first year" requires knowing 365 days (time unit not stated)

requires_planning: true - dependencies exist throughout the chain

num_contexts: 2 - "first 180 days" and "remaining days" are separate periods

intermediate_results_used: 4 - result 180 used in step 4, result 185 used in step 3, result 370 used in step 4, result 550 used in step 5

max_chain_depth: 4 - longest path is step 2-3-4-5 (length 4)

num_entities: 5 - cups-period1, days-remaining, cups-period2, total-cups, bags

**This format allows the validator to check your reasoning for each field.**

---

# VALIDATION CHECKLIST

Before submitting, verify:
- Exactly 16 fields present
- `len(ops) == steps`
- All boolean values are `true` or `false` (lowercase, no quotes)
- `div_exact` is `null` (no quotes) if no division exists
- Valid JSON syntax (double quotes, proper commas, no trailing comma)
- All integer fields have numeric values (no quotes)
- `ops` array contains only: "add", "sub", "mult", "div"

---

# COMPLETE EXAMPLE

**This example shows the REQUIRED output format: both JSON annotation AND reasoning explanations.**

---**Problem:**
"Cecilia just bought a new puppy. According to her veterinarian, she has to feed the puppy 1 cup of dog food every day for the first 180 days. Then she has to feed the puppy 2 cups of dog food every day for the rest of its life. If one bag of dog food contains 110 cups, how many bags of dog food will Cecilia use in the first year?"

**Answer:**
"The number of cups of dog food given for the first 180 days is 180 * 1 = 180 cups. The number of days left in the year is 365 - 180 = 185. The number of cups of dog food given for the rest of the first year is 185 * 2 = 370 cups. The total number of cups given in the first year is 180 + 370 = 550 cups. The number of bags of dog food used in the first year is 550 / 110 = 5 bags."

**Correct Annotation:**
```json
{
  "ops": ["mult", "sub", "mult", "add", "div"],
  "steps": 5,
  "can_parallelize": false,
  "max_num": 550,
  "max_operand_digits": 3,
  "has_decimals": false,
  "has_carrying": true,
  "has_borrowing": false,
  "div_exact": true,
  "has_rate_or_ratio": true,
  "requires_world_knowledge": true,
  "requires_planning": true,
  "num_contexts": 2,
  "intermediate_results_used": 3,
  "max_chain_depth": 4,
  "num_entities": 5
}
```

**Detailed Reasoning:**

### TIER 1 ANALYSIS (Direct Probing Features)

**1. ops: ["mult", "sub", "mult", "add", "div"]**

*Why:* Reading through the solution step by step:
- Step 1: "180 * 1 = 180" is multiplication
- Step 2: "365 - 180 = 185" is subtraction  
- Step 3: "185 * 2 = 370" is multiplication
- Step 4: "180 + 370 = 550" is addition
- Step 5: "550 / 110 = 5" is division

Listed in execution order, giving us 5 operations.

---

**2. steps: 5**

*Why:* We counted 5 operations above, so steps = 5. This must always equal the length of the `ops` array.

---

**3. can_parallelize: false**

*Why:* Let me check if any operations can be done simultaneously:
- Step 1 (180*1) is independent
- Step 2 (365-180) needs the result 180 from step 1 (no)
- Step 3 (185*2) needs the result 185 from step 2 (no)
- Step 4 (180+370) needs results from steps 1 and 3 (no)
- Step 5 (550/110) needs the result 550 from step 4 (no)

Every step depends on previous results, forming a pure sequential chain. Therefore `false`.

---

**4. max_num: 550**

*Why:* Scanning all numbers that appear:
- From problem: 1, 180, 2, 365, 110
- From calculations: 180, 185, 370, 550, 5

The largest is 550.

---

**5. max_operand_digits: 3**

*Why:* Checking digit counts for all operands in all operations:
- 180 * 1: [3, 1]
- 365 - 180: [3, 3]
- 185 * 2: [3, 1]
- 180 + 370: [3, 3]
- 550 / 110: [3, 3]

Maximum digit count across all operands is 3.

---

**6. has_decimals: false**

*Why:* Checking all numbers in the problem:
- All inputs are integers: 1, 180, 2, 365, 110
- All intermediate results are integers: 180, 185, 370, 550, 5
- No fractions, no decimal points, no percentages

Therefore `false`.

---

**7. has_carrying: true**

*Why:* Checking each addition operation:

Only one addition exists: 180 + 370

```
  180
+ 370
-----
```
- Ones place: 0 + 0 = 0 (no carry)
- Tens place: 8 + 7 = 15 (>=10, REQUIRES CARRYING!)
- Hundreds place: 1 + 3 + 1(carry) = 5

Since the tens place requires carrying, `has_carrying = true`.

---

**8. has_borrowing: true**

*Why:* Checking each subtraction operation:

Only one subtraction exists: 365 - 180

```
  365
- 180
-----
```
- Ones place: 5 - 0 = 5 (ok, no borrow needed)
- Tens place: 6 - 8 = ? (6 < 8, REQUIRES BORROWING!)
  - Must borrow from hundreds: 16 - 8 = 8
- Hundreds place: 2 - 1 = 1 (after lending to tens)

Since the tens place requires borrowing, `has_borrowing = true`.

---

**9. div_exact: true**

*Why:* Checking the division operation:
- 550 / 110 = 5 exactly (no remainder)
- 550 = 110 * 5 with remainder 0

Since division is exact, `div_exact = true`.

---

**10. has_rate_or_ratio: true**

*Why:* Looking for rates/ratios in the problem text:
- "1 cup of dog food every day" → rate (cups per day)
- "2 cups of dog food every day" → rate (cups per day)

The problem explicitly uses rates, so `has_rate_or_ratio = true`.

---

**11. requires_unit_conversion: true**

*Why:* Checking what conversions are needed:
- The problem mentions "the first year"
- The solution uses "365" for days in a year
- This requires converting from "year" (time unit) to "days" (different time unit)
- The conversion factor (365 days = 1 year) is not stated in the problem
- This is a unit conversion that must be known

Therefore `requires_unit_conversion = true`.

---

### TIER 2 ANALYSIS (Structural Features)

**12. requires_planning: true**

*Why:* Since we already established `can_parallelize = false` and `steps = 5 ≥ 3`, this indicates dependencies exist. Let me verify:
- To compute step 2, I need step 1's result (180)
- To compute step 3, I need step 2's result (185)  
- To compute step 4, I need results from steps 1 and 3
- To compute step 5, I need step 4's result (550)

Clear dependencies exist, so execution must be planned. Therefore `requires_planning = true`.

---

**13. num_contexts: 2**

*Why:* Identifying distinct scenarios/time periods:
- **Context 1:** "first 180 days" - separate calculation period
- **Context 2:** "rest of its life" / "rest of the first year" (remaining 185 days) - different calculation period

The problem splits the year into two distinct time periods that are calculated separately, then combined. Therefore `num_contexts = 2`.

---

**14. intermediate_results_used: 3**

*Why:* Tracking which intermediate results get reused:

Step 1 produces: 180
- Used in: Step 4 (180 + 370) REUSED

Step 2 produces: 185  
- Used in: Step 3 (185 * 2) REUSED

Step 3 produces: 370
- Used in: Step 4 (180 + 370) REUSED

Step 4 produces: 550
- Used in: Step 5 (550 / 110) REUSED

Step 5 produces: 5
- This is the final answer, not reused

Counting reused intermediate results: 180, 185, 370, 550 = 4 total

**CORRECTION:** `intermediate_results_used = 4` (I initially miscounted - all four intermediate results are reused)

---

**15. max_chain_depth: 4**

*Why:* Building the dependency graph and finding longest path:

```
Step 1 (180*1=180): no dependencies, depth 1
Step 2 (365-180=185): depends on step 1, depth 2
Step 3 (185*2=370): depends on step 2, depth 3
Step 4 (180+370=550): depends on steps 1 and 3, depth max(1,3)+1 = 4
Step 5 (550/110=5): depends on step 4, depth 5
```

Wait, let me recalculate more carefully:
- Path 1-4-5: length 3
- Path 2-3-4-5: length 4 (LONGEST PATH)

The longest chain from start to finish is 4 steps deep. Therefore `max_chain_depth = 4`.

---

**16. num_entities: 5**

*Why:* Identifying each distinct quantity being tracked:

1. **Cups for first period** (180): amount of food in first 180 days
2. **Days remaining** (185): time left in the year - different type of quantity
3. **Cups for second period** (370): amount of food in remaining days
4. **Total cups** (550): aggregate of cups from both periods
5. **Number of bags** (5): final answer in different units (bags not cups)

Each represents a conceptually different quantity. Therefore `num_entities = 5`.

**CORRECTED Final Annotation:**
```json
{
  "ops": ["mult", "sub", "mult", "add", "div"],
  "steps": 5,
  "can_parallelize": false,
  "max_num": 550,
  "max_operand_digits": 3,
  "has_decimals": false,
  "has_carrying": true,
  "has_borrowing": true,
  "div_exact": true,
  "has_rate_or_ratio": true,
  "requires_unit_conversion": true,
  "requires_planning": true,
  "num_contexts": 2,
  "intermediate_results_used": 4,
  "max_chain_depth": 4,
  "num_entities": 5
}
```

---

# COMMON MISTAKES TO AVOID

**Don't** output ONLY the JSON without reasoning explanations
**Don't** forget to check carrying/borrowing manually  
**Don't** count final answer as an intermediate result
**Don't** confuse contexts with entities (contexts are scenarios, entities are quantities)
**Don't** guess on Tier 2 fields - trace through the logic carefully
**Don't** use quotes around boolean values in JSON (`true` not `"true"`)
**Don't** use quotes around null in JSON (`null` not `"null"`)

**DO** provide both JSON and reasoning for each field
**DO** show your work for carrying/borrowing checks
**DO** explain why you chose each value

---

Now annotate the provided problem following these instructions exactly. 

**Output both:**
1. The JSON annotation
2. Your reasoning for each field (so the validator can check your work)
"""

# Compare function
def compare_annotations(expected, actual):
    """ops checked as sets, rest by equality."""
    if len(expected.get('ops', [])) != len(actual.get('ops', [])):
        return False
    if sorted(expected.get('ops', [])) != sorted(actual.get('ops', [])):
        return False
    for key in expected:
        if key == 'ops':
            continue
        if expected[key] != actual.get(key):
            return False
    return True


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Annotate GSM8K dataset examples')
    parser.add_argument('-n', '--num-examples', type=int, default=DEFAULT_NUM_EXAMPLES,
                        help=f'Number of examples to process (default: {DEFAULT_NUM_EXAMPLES})')
    parser.add_argument('-s', '--split', type=str, default=DEFAULT_SPLIT,
                        choices=['train', 'test'],
                        help=f'Dataset split to use: train or test (default: {DEFAULT_SPLIT})')
    parser.add_argument('-o', '--output', type=str, default='annotation/annotations.json',
                        help='Output file path for annotations (default: annotation/annotations.json)')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Starting index in the dataset (default: 0)')
    
    args = parser.parse_args()
    
    # Load the dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    dataset = load_from_disk(DATASET_PATH)
    
    # Get the specified split
    split_data = dataset[args.split]
    print(f"Using '{args.split}' split with {len(split_data)} total examples")
    
    # Calculate end index
    end_idx = min(args.start_idx + args.num_examples, len(split_data))
    actual_num = end_idx - args.start_idx
    print(f"Processing {actual_num} examples (indices {args.start_idx} to {end_idx-1})")
    
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    annotations = []
    
    for idx in range(args.start_idx, end_idx):
        example = split_data[idx]
        print(f"\nProcessing example {idx + 1}/{end_idx} (index {idx})")
        
        user_prompt = f"""**Question:**

{example['question']}

**Answer:**
{example['answer']}"""
        
        # Make request
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=8000,
                system=[{
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"}
                }],
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            response_text = response.content[0].text
            print(response_text)
            
            # Store annotation with metadata
            annotation_entry = {
                "index": idx,
                "split": args.split,
                "question": example['question'],
                "answer": example['answer'],
                "annotation_response": response_text
            }
            annotations.append(annotation_entry)
            
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            annotation_entry = {
                "index": idx,
                "split": args.split,
                "question": example['question'],
                "answer": example['answer'],
                "error": str(e)
            }
            annotations.append(annotation_entry)
    
    print(f"\nSaving {len(annotations)} annotations to {args.output}")
    
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(args.output, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Done! Annotations saved to {args.output}")


if __name__ == "__main__":
    main()