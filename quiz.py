import json
import argparse
import subprocess
import random
import tempfile
import os
import re
from typing import List, Dict, Any, Tuple
import math

def load(input_file: str) -> List[Dict[str, str]]:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("qa_pairs", [])

def batch(qa_pairs: List[Dict[str, str]], batch_size: int) -> List[List[Dict[str, str]]]:
    return [qa_pairs[i:i + batch_size] for i in range(0, len(qa_pairs), batch_size)] #divide questions into batches to avoid overwhelming LLM

def prompt(questions: List[Dict[str, str]], all_answers: List[str]) -> str:
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
    
    prompt += "\n## Mulige svar (disse er stokket):\n"
    for i, answer in enumerate(shuffled_answers, 1):
        prompt += f"{i}. {answer}\n"
        
    prompt += "\nVelg det beste svaret for hvert spørsmål. Husk at hvert svar kan kun brukes én gang."
    prompt += "\nDu MÅ kopiere svarene NØYAKTIG og FULLSTENDIG slik de vises i listen - inkluder ALL informasjon, også i parentes."
    
    return prompt, shuffled_answers

def run_command(command: str, prompt: str) -> str:
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

def extract_json(response: str) -> Dict[str, Any]:
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
    try:
        import re
        json_pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}'
        match = re.search(json_pattern, response)
        if match:
            return json.loads(match.group(0))
    except (json.JSONDecodeError, re.error):
        pass
    
    print("Could not extract JSON")
    print("Raw response:", response)
    return {"svar": []}

def normalize(text: str) -> str:
    text = re.sub(r'^\d+\.\s*', '', text)
    text = re.sub(r'[.,;:!?]+$', '', text)
    return text.strip().lower()

def clean(text: str) -> str:
    return re.sub(r'^\d+\.\s*', '', text)

def process(batch_questions: List[Dict[str, str]], 
                     batch_responses: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    results = []
    
    question_to_correct_answer = {
        clean(qa['question']): qa['answer'] 
        for qa in batch_questions
    }
    
    normalized_correct_answers = {
        q: normalize(a) 
        for q, a in question_to_correct_answer.items()
    }
    
    for response_item in batch_responses:
        raw_question = response_item.get("spørsmål", "")
        clean_q = clean(raw_question)
        selected_answer = response_item.get("valgt_svar", "")
        
        correct_answer = question_to_correct_answer.get(clean_q, "")
        
        normalized_selected = normalize(selected_answer)
        normalized_correct = normalized_correct_answers.get(clean_q, "")
        
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
    random.seed(args.seed)
    qa_pairs = load(args.input)
    
    if args.num_questions is not None:
        qa_pairs = qa_pairs[:args.num_questions]
    
    all_answers = [qa['answer'] for qa in qa_pairs]
    batches = batch(qa_pairs, args.batch_size)
    
    all_results = []
    total_correct = 0
    
    print(f"Processing {len(qa_pairs)} questions in {len(batches)} batches...")
    
    for i, batch in enumerate(batches, 1):
        print(f"Processing batch {i}/{len(batches)}...")
        
        prompt, shuffled_answers = prompt(batch, all_answers)
        llm_response = run_command(args.llm, prompt)
        
        parsed_response = extract_json(llm_response)
        batch_responses = parsed_response.get("svar", [])
        
        batch_results = process(batch, batch_responses)
        all_results.extend(batch_results)
        
        batch_correct = sum(1 for r in batch_results if r.get("is_correct", False))
        total_correct += batch_correct
        
        print(f"Batch {i} results: {batch_correct}/{len(batch)} correct")
    
    accuracy = total_correct/len(qa_pairs) if qa_pairs else 0
    
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
    
    #include normalized answers if desired
    if args.debug:
        for i, r in enumerate(all_results):
            final_results["results"][i].update({
                "normalized_selected": r.get("normalized_selected", ""),
                "normalized_correct": r.get("normalized_correct", "")
            })
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Final accuracy: {total_correct}/{len(qa_pairs)} ({accuracy:.2%})")

if __name__ == "__main__":
    main()
