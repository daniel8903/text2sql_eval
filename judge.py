import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv, dotenv_values

load_dotenv()
_cfg = dotenv_values()

class SiemensAPIClient:
    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=_cfg["SIEMENS_API_KEY"],
            base_url=_cfg["SIEMENS_API_BASE"],
        )

    def chat(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        stream: bool = False,
    ) -> Dict[str, Any]:
        rsp = self._client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stream=stream,
        )
        first_choice = rsp.choices[0].message.content
        return {
            "message": {"content": first_choice},
            "usage": getattr(rsp, "usage", None),
        }

class EnhancedSQLJudge:
    def __init__(self, model_name, api_client=None):
        """Initialize the judge with the specified model."""
        self.model_name = model_name
        self.api_client = api_client if api_client else SiemensAPIClient()

    def evaluate_sql_pair(self, prompt, ref_sql, gen_sql, context):
        """Evaluate SQL pair using the enhanced judgment criteria."""
        system_prompt = (
            "You are an expert SQL analyst with deep understanding of business intelligence queries. "
            "Your task is to comprehensively evaluate SQL queries beyond simple equivalence. "
            "Consider the following aspects:\n"
            "1. Technical correctness (syntax, logic)\n"
            "2. Semantic equivalence (would produce same results)\n"
            "3. Business intent fulfillment (addresses the actual question asked)\n"
            "4. Query efficiency and elegance\n"
            "Respond with a JSON object containing:\n"
            "  'technically_equivalent': boolean (would produce identical results),\n"
            "  'fulfills_intent': boolean (correctly answers the business question),\n"
            "  'superiority': string ('reference', 'generated', or 'equal'),\n"
            "  'explanation': string (detailed analysis),\n"
            "  'overall_assessment': string ('correct', 'incorrect', or 'differently_correct')"
        )
        
        user_prompt = (
            "### Original Natural Language Question\n"
            f"{prompt}\n\n"
            "### Context (database schema)\n"
            f"{context}\n\n"
            "### Reference SQL\n"
            f"{ref_sql}\n\n"
            "### Generated SQL\n"
            f"{gen_sql}\n\n"
            "### Task\n"
            "Evaluate these queries considering technical equivalence, fulfillment of business intent, "
            "and overall correctness. Which query better answers the original question? "
            "Return your assessment in the requested JSON format."
        )

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.api_client.chat(
                model_name=self.model_name,
                messages=messages,
                temperature=0.1,
                stream=False
            )
            
            result = response["message"]["content"]
            
            # Extract JSON from response
            json_content = self._extract_json(result)
            assessment = json.loads(json_content)
            
            # Ensure all fields are present
            return {
                "technically_equivalent": bool(assessment.get("technically_equivalent", False)),
                "fulfills_intent": bool(assessment.get("fulfills_intent", False)),
                "superiority": str(assessment.get("superiority", "equal")),
                "explanation": str(assessment.get("explanation", "")),
                "overall_assessment": str(assessment.get("overall_assessment", "incorrect"))
            }
            
        except Exception as e:
            print(f"Error evaluating SQL: {e}")
            return {
                "technically_equivalent": False,
                "fulfills_intent": False,
                "superiority": "reference",
                "explanation": f"Evaluation error: {str(e)}",
                "overall_assessment": "incorrect"
            }
    
    def _extract_json(self, text):
        """Extract JSON content from the response text."""
        # Look for JSON between curly braces
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end >= 0:
            return text[start:end+1]
        return "{}"

def process_file(input_file, output_file, judge):
    """Process a JSONL file and add enhanced judgments."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"Input file {input_file} does not exist.")
        return
    
    examples = []
    with open(input_path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Processing {len(examples)} examples...")
    
    # Open output file once and keep it open
    with open(output_path, 'w') as out_file:
        for i, example in enumerate(tqdm(examples)):
            # Extract necessary information
            prompt = example["prompt"]
            context = example["context"]
            ref_sql = example["reference_sql"]
            gen_sql = example["generated_sql"]
            
            # Get enhanced judgment
            judgment = judge.evaluate_sql_pair(prompt, ref_sql, gen_sql, context)
            
            # Add judgment to the example
            example["enhanced_judgment"] = judgment
            
            # Write this example immediately to the output file
            out_file.write(json.dumps(example) + '\n')
            out_file.flush()  # Force write to disk
            
            # Optional progress update
            if (i + 1) % 5 == 0 or i == len(examples) - 1:
                print(f"Processed {i + 1}/{len(examples)} examples")
    
    print(f"Enhanced judgments added. Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Add enhanced SQL judgments to JSONL files")
    parser.add_argument("input_file", help="Path to input JSONL file")
    parser.add_argument("output_file", help="Path to output JSONL file")
    parser.add_argument("--model", default="qwen3-30b", help="Model to use for judgments")
    
    args = parser.parse_args()
    
    # Create API client
    api_client = SiemensAPIClient()
    
    # Create judge with Qwen3 30B model
    judge = EnhancedSQLJudge(args.model, api_client)
    process_file(args.input_file, args.output_file, judge)

if __name__ == "__main__":
    main()
