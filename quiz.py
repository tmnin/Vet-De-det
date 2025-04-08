#!/usr/bin/env python3
import json
import argparse
import random
import os
import re
from typing import List, Dict, Any
import requests
import time
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

def run_groq_api(api_key: str, model: str, prompt: str, max_retries: int = 5, initial_retry_delay: int = 5) -> str:
    """Execute query using Groq API and return the response with robust rate limit handling."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,  # Low temperature for more deterministic answers
        "max_tokens": 4000
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data)
            
            # Handle rate limiting (429)
            if response.status_code == 429:
                # Get retry-after header or use exponential backoff
                retry_after = int(response.headers.get('Retry-After', initial_retry_delay * (2 ** attempt)))
                print(f"Rate limit hit. Waiting for {retry_after} seconds before retry...")
                time.sleep(retry_after)
                continue
                
            # Handle other errors
            if response.status_code != 200:
                print(f"API request failed with status code {response.status_code}: {response.text}")
                if attempt < max_retries - 1:
                    retry_delay = initial_retry_delay * (2 ** attempt)
                    print(f"Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    response.raise_for_status()
            
            return response.json()["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            retry_delay = initial_retry_delay * (2 ** attempt)
            print(f"API request failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Giving up on this batch.")
                raise

def save_progress(results, total_correct, total_processed, output_file, model_name):
    """Save current progress to a temporary file."""
    progress_file = f"{output_file}.progress"
    
    # Calculate current accuracy
    accuracy = total_correct / total_processed if total_processed else 0
    
    # Prepare progress results
    progress_results = {
        "results": results,
        "total_questions_processed": total_processed,
        "correct_answers": total_correct,
        "current_accuracy": accuracy,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save progress
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_results, f, ensure_ascii=False, indent=2)
    
    print(f"Progress saved to {progress_file}")

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
    # Store exactly as they appear in the original data to preserve formatting
    question_to_answer = {}
    for qa in batch_questions:
        clean_q = clean_question(qa['question'])
        question_to_answer[clean_q] = qa['answer']
    
    # Also create a case-insensitive mapping for better matching
    case_insensitive_map = {q.lower(): a for q, a in question_to_answer.items()}
    
    # Create normalized versions of correct answers for comparison
    normalized_correct_answers = {q: normalize_answer(a) for q, a in question_to_answer.items()}
    
    for response_item in batch_responses:
        # Get the question (might have numbers at the beginning)
        raw_question = response_item.get("spørsmål", "")
        
        # Clean the question to match our dictionary keys
        clean_q = clean_question(raw_question)
        
        selected_answer = response_item.get("valgt_svar", "")
        
        # Find correct answer for this question - try exact match first
        correct_answer = question_to_answer.get(clean_q, "")
        
        # If not found, try case-insensitive match
        if not correct_answer:
            lower_q = clean_q.lower()
            if lower_q in case_insensitive_map:
                correct_answer = case_insensitive_map[lower_q]
            else:
                # Try partial matching as a last resort
                for q in question_to_answer:
                    if clean_q in q or q in clean_q:
                        correct_answer = question_to_answer[q]
                        break
        
        # Normalize both answers for comparison
        normalized_selected = normalize_answer(selected_answer)
        normalized_correct = normalize_answer(correct_answer)
        
        # Check if the normalized answer is correct
        is_correct = normalized_selected == normalized_correct
        
        results.append({
            "question": raw_question,
            "selected_answer": selected_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "normalized_selected": normalized_selected,
            "normalized_correct": normalized_correct
        })
    
    return results

def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    """Load checkpoint data if it exists."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Checkpoint file {checkpoint_file} is corrupted. Starting fresh.")
    return None

def main():
    parser = argparse.ArgumentParser(description="Quiz LLMs on Norwegian history questions using Groq API")
    parser.add_argument("--input", required=True, help="Input JSON file with QA pairs")
    parser.add_argument("--output", required=True, help="Output JSON file for results")
    parser.add_argument("--num-questions", type=int, default=None, 
                        help="Number of questions to ask (default: all)")
    parser.add_argument("--batch-size", type=int, default=5, 
                        help="Number of questions per batch (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--debug", action="store_true", help="Include debug information in output")
    parser.add_argument("--api-key", help="Groq API key (can also be set via GROQ_API_KEY env var)")
    parser.add_argument("--model", default="llama3-70b-8192", 
                        help="Groq model to use (default: llama3-70b-8192)")
    parser.add_argument("--checkpoint", action="store_true", 
                        help="Enable checkpointing to resume from interruptions")
    parser.add_argument("--checkpoint-interval", type=int, default=5,
                        help="Save checkpoint every N batches (default: 5)")
    parser.add_argument("--rate-limit-delay", type=int, default=10,
                        help="Base delay in seconds between API requests (default: 10)")
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise ValueError("API key must be provided via --api-key argument or GROQ_API_KEY environment variable")
    
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
    
    # Check for checkpoint
    checkpoint_file = f"{args.output}.checkpoint"
    checkpoint_data = None
    if args.checkpoint:
        checkpoint_data = load_checkpoint(checkpoint_file)
    
    # Initialize or restore progress
    all_results = []
    total_correct = 0
    start_batch = 0
    
    if checkpoint_data:
        all_results = checkpoint_data.get("results", [])
        total_correct = checkpoint_data.get("correct_answers", 0)
        start_batch = checkpoint_data.get("last_batch_processed", 0) + 1
        print(f"Resuming from checkpoint: {len(all_results)} questions processed, {total_correct} correct answers")
    
    print(f"Processing {len(qa_pairs)} questions in {len(batches)} batches using Groq API...")
    print(f"Model: {args.model}")
    print(f"Starting from batch: {start_batch+1}")
    
    # Create a full question-to-answer map for verification at the end
    full_question_map = {}
    for qa in qa_pairs:
        clean_q = clean_question(qa['question'])
        full_question_map[clean_q] = qa['answer']
    
    for i, batch in enumerate(batches[start_batch:], start_batch+1):
        print(f"Processing batch {i}/{len(batches)}...")
        
        # Delay between batches to avoid rate limiting
        if i > start_batch+1:  # Don't delay the first batch
            delay = args.rate_limit_delay
            print(f"Waiting {delay} seconds before next batch...")
            time.sleep(delay)
        
        # Create prompt for this batch
        prompt, shuffled_answers = create_prompt(batch, all_answers)
        
        try:
            # Run LLM via Groq API with retries
            llm_response = run_groq_api(api_key, args.model, prompt, 
                                         max_retries=5, initial_retry_delay=args.rate_limit_delay)
            
            # Parse response
            parsed_response = extract_json_from_response(llm_response)
            batch_responses = parsed_response.get("svar", [])
            
            # Process results
            batch_results = process_responses(batch, batch_responses)
            
            # Verify all questions have correct answers
            for result in batch_results:
                clean_q = clean_question(result["question"])
                if not result["correct_answer"]:
                    # If still missing, do a direct lookup from the full map
                    result["correct_answer"] = full_question_map.get(clean_q, "")
                    # Re-evaluate if correct with the updated correct answer
                    if result["correct_answer"]:
                        result["normalized_correct"] = normalize_answer(result["correct_answer"])
                        result["is_correct"] = result["normalized_selected"] == result["normalized_correct"]
            
            all_results.extend(batch_results)
            
            # Count correct answers
            batch_correct = sum(1 for r in batch_results if r.get("is_correct", False))
            total_correct += batch_correct
            
            print(f"Batch {i} results: {batch_correct}/{len(batch)} correct")
            
            # Save checkpoint if enabled
            if args.checkpoint and i % args.checkpoint_interval == 0:
                checkpoint = {
                    "results": all_results,
                    "correct_answers": total_correct,
                    "last_batch_processed": i-1,
                    "total_batches": len(batches),
                    "model": args.model,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint, f, ensure_ascii=False, indent=2)
                
                print(f"Checkpoint saved after batch {i}")
                
                # Also save current progress to the output file
                save_progress(
                    all_results, 
                    total_correct, 
                    len(all_results), 
                    args.output,
                    args.model
                )
                
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            # Save progress before exit if checkpointing is enabled
            if args.checkpoint:
                checkpoint = {
                    "results": all_results,
                    "correct_answers": total_correct,
                    "last_batch_processed": i-2,  # Mark the last successfully processed batch
                    "total_batches": len(batches),
                    "failed_at_batch": i-1,
                    "model": args.model,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint, f, ensure_ascii=False, indent=2)
                
                print(f"Checkpoint saved after error in batch {i}")
                
                # Also save current progress
                save_progress(
                    all_results, 
                    total_correct, 
                    len(all_results), 
                    args.output,
                    args.model
                )
                
            # Continue with next batch
            continue
    
    # Final verification pass - ensure all questions have correct answers
    questions_with_missing_answers = []
    for result in all_results:
        if not result["correct_answer"]:
            clean_q = clean_question(result["question"])
            questions_with_missing_answers.append(clean_q)
    
    if questions_with_missing_answers:
        print(f"WARNING: {len(questions_with_missing_answers)} questions still have missing answers:")
        for q in questions_with_missing_answers[:5]:  # Show first 5
            print(f"  - {q}")
        if len(questions_with_missing_answers) > 5:
            print(f"  - ... and {len(questions_with_missing_answers) - 5} more")
    
    # Calculate overall accuracy
    accuracy = total_correct / len(all_results) if all_results else 0
    
    # Prepare final results
    final_results = {
        "results": [{
            "question": r["question"],
            "selected_answer": r["selected_answer"],
            "correct_answer": r["correct_answer"],
            "is_correct": r["is_correct"]
        } for r in all_results],
        "total_questions_processed": len(all_results),
        "total_questions_in_dataset": len(qa_pairs),
        "correct_answers": total_correct,
        "accuracy": accuracy,
        "model": args.model
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
    
    # Clean up checkpoint file if processing completed successfully
    if args.checkpoint and os.path.exists(checkpoint_file) and len(all_results) == len(qa_pairs):
        try:
            os.remove(checkpoint_file)
            print(f"Checkpoint file removed after successful completion")
        except:
            print(f"Note: Could not remove checkpoint file {checkpoint_file}")
    
    print(f"\nResults saved to {args.output}")
    print(f"Final accuracy: {total_correct}/{len(all_results)} ({accuracy:.2%})")
    if len(all_results) < len(qa_pairs):
        print(f"Note: Only processed {len(all_results)} of {len(qa_pairs)} total questions")

if __name__ == "__main__":
    main()