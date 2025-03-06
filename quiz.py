#!/usr/bin/env python3
import json
import argparse
import subprocess
import random
import tempfile
import os
import re
from typing import List, Dict, Any, Tuple
import math

def load_qa_pairs(input_file: str) -> List[Dict[str, str]]:
    """Load question-answer pairs from JSON file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("qa_pairs", [])

def create_batches(qa_pairs: List[Dict[str, str]], batch_size: int) -> List[List[Dict[str, str]]]:
    """Split the question-answer pairs into batches."""
    return [qa_pairs[i:i + batch_size] for i in range(0, len(qa_pairs), batch_size)]

def create_prompt(questions: List[Dict[str, str]], all_answers: List[str]) -> str:
    """Create a Norwegian prompt for the LLM."""
    shuffled_answers = all_answers.copy()
    random.shuffle(shuffled_answers)
    
    prompt = """
# Norsk Historie Quiz

Du vil bli presentert med spørsmål om norsk historie fra rundt 1940 og en liste med mulige svar.
Din oppgave er å matche hvert spørsmål med det mest korrekte svaret fra listen.

VIKTIGE REGLER:
1. Hvert svar kan kun brukes én gang.
2. For hvert spørsmål, velg ett svar fra listen.
3. Du MÅ returnere svaret NØYAKTIG og FULLSTENDIG slik det står i listen - inkluder ALL informasjon i parentes og eksempler.
4. Ikke legg til nummering eller endre teksten på noen måte.
5. Returner svaret ditt i JSON-format som følger:
```json
{
  "svar": [
    {
      "spørsmål": "Spørsmål 1",
      "valgt_svar": "Ditt valgte svar for spørsmål 1 (inkludert all informasjon i parentes)"
    },
    {
      "spørsmål": "Spørsmål 2",
      "valgt_svar": "Ditt valgte svar for spørsmål 2 (inkludert all informasjon i parentes)"
    }
    ...
  ]
}
```

## Spørsmål:
"""
    
    for i, qa in enumerate(questions, 1):
        # Strip numbers from the beginning of questions
        clean_question = re.sub(r'^\d+\.\s*', '', qa['question'])
        prompt += f"{i}. {clean_question}\n"
    
    prompt += "\n## Mulige svar (disse er stokket):\n"
    for i, answer in enumerate(shuffled_answers, 1):
        prompt += f"{i}. {answer}\n"
        
    prompt += "\nVelg det beste svaret for hvert spørsmål. Husk at hvert svar kan kun brukes én gang."
    prompt += "\nDu MÅ kopiere svarene NØYAKTIG og FULLSTENDIG slik de vises i listen - inkluder ALL informasjon, også i parentes."
    
    return prompt, shuffled_answers

def run_llm_command(command: str, prompt: str) -> str:
    """Execute LLM command and return the output."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as temp:
        temp.write(prompt)
        temp_path = temp.name
    
    try:
        result = subprocess.run(
            f"{command} < {temp_path}", 
            shell=True, 
            capture_output=True, 
            text=True
        )
        return result.stdout
    finally:
        os.unlink(temp_path)

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """Extract JSON from the LLM response."""
    # Find json block between ```json and ```
    json_start = response.find("```json")
    if json_start == -1:
        json_start = response.find("```")
    
    if json_start != -1:
        json_start = response.find("{", json_start)
        json_end = response.rfind("}")
        if json_start != -1 and json_end != -1:
            try:
                return json.loads(response[json_start:json_end+1])
            except json.JSONDecodeError:
                pass
    
    # Try to find any JSON object in the response
    try:
        import re
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        match = re.search(json_pattern, response)
        if match:
            return json.loads(match.group(0))
    except (json.JSONDecodeError, re.error):
        pass
    
    # If all else fails, return empty dict
    print("Warning: Could not extract JSON from response")
    print("Raw response:", response)
    return {"svar": []}

def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    # Remove leading numbers (like "13. ")
    text = re.sub(r'^\d+\.\s*', '', text)
    
    # Remove trailing punctuation
    text = re.sub(r'[.,;:!?]+$', '', text)
    
    # Strip whitespace and convert to lowercase
    return text.strip().lower()

def clean_question(text: str) -> str:
    """Remove numbering from the beginning of questions."""
    return re.sub(r'^\d+\.\s*', '', text)

def process_responses(batch_questions: List[Dict[str, str]], 
                     batch_responses: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Process LLM responses and compute accuracy."""
    results = []
    
    # Create a mapping from questions to their correct answers
    # Clean questions by removing leading numbers
    question_to_correct_answer = {
        clean_question(qa['question']): qa['answer'] 
        for qa in batch_questions
    }
    
    # Create normalized versions of correct answers for comparison
    normalized_correct_answers = {
        q: normalize_answer(a) 
        for q, a in question_to_correct_answer.items()
    }
    
    for response_item in batch_responses:
        # Get the question (might have numbers at the beginning)
        raw_question = response_item.get("spørsmål", "")
        
        # Clean the question to match our dictionary keys
        clean_q = clean_question(raw_question)
        
        selected_answer = response_item.get("valgt_svar", "")
        
        # Find correct answer for this question
        correct_answer = question_to_correct_answer.get(clean_q, "")
        
        # Normalize both answers for comparison
        normalized_selected = normalize_answer(selected_answer)
        normalized_correct = normalized_correct_answers.get(clean_q, "")
        
        # Check if the normalized answer is correct
        is_correct = normalized_selected == normalized_correct
        
        results.append({
            "question": raw_question,
            "selected_answer": selected_answer,
            "correct_answer": correct_answer,
            "normalized_selected": normalized_selected,
            "normalized_correct": normalized_correct,
            "is_correct": is_correct
        })
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Quiz LLMs on Norwegian history questions")
    parser.add_argument("--input", required=True, help="Input JSON file with QA pairs")
    parser.add_argument("--llm", required=True, help="LLM command to run")
    parser.add_argument("--output", required=True, help="Output JSON file for results")
    parser.add_argument("--num-questions", type=int, default=None, 
                        help="Number of questions to ask (default: all)")
    parser.add_argument("--batch-size", type=int, default=10, 
                        help="Number of questions per batch (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--debug", action="store_true", help="Include debug information in output")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load QA pairs
    qa_pairs = load_qa_pairs(args.input)
    
    # Limit to specified number of questions if provided
    if args.num_questions is not None:
        qa_pairs = qa_pairs[:args.num_questions]
    
    # Extract all answers
    all_answers = [qa['answer'] for qa in qa_pairs]
    
    # Split into batches
    batches = create_batches(qa_pairs, args.batch_size)
    
    all_results = []
    total_correct = 0
    
    print(f"Processing {len(qa_pairs)} questions in {len(batches)} batches...")
    
    for i, batch in enumerate(batches, 1):
        print(f"Processing batch {i}/{len(batches)}...")
        
        # Create prompt for this batch
        prompt, shuffled_answers = create_prompt(batch, all_answers)
        
        # Run LLM
        llm_response = run_llm_command(args.llm, prompt)
        
        # Parse response
        parsed_response = extract_json_from_response(llm_response)
        batch_responses = parsed_response.get("svar", [])
        
        # Process results
        batch_results = process_responses(batch, batch_responses)
        all_results.extend(batch_results)
        
        # Count correct answers
        batch_correct = sum(1 for r in batch_results if r.get("is_correct", False))
        total_correct += batch_correct
        
        print(f"Batch {i} results: {batch_correct}/{len(batch)} correct")
    
    # Calculate overall accuracy
    accuracy = total_correct / len(qa_pairs) if qa_pairs else 0
    
    # Prepare final results
    final_results = {
        "results": [{
            "question": r["question"],
            "selected_answer": r["selected_answer"],
            "correct_answer": r["correct_answer"],
            "is_correct": r["is_correct"]
        } for r in all_results],
        "total_questions": len(qa_pairs),
        "correct_answers": total_correct,
        "accuracy": accuracy
    }
    
    # Include debug information if requested
    if args.debug:
        for i, r in enumerate(all_results):
            final_results["results"][i].update({
                "normalized_selected": r.get("normalized_selected", ""),
                "normalized_correct": r.get("normalized_correct", "")
            })
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Final accuracy: {total_correct}/{len(qa_pairs)} ({accuracy:.2%})")

if __name__ == "__main__":
    main()