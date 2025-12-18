"""Shared utilities for evaluation scripts.

This module contains common functions used across multiple evaluation scripts
to reduce code duplication and improve maintainability.
"""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import openai
from dotenv import load_dotenv

load_dotenv()

# Judge settings
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 500


def clean_thinking_blocks(text: str) -> str:
    """Remove thinking/reasoning blocks from model responses.
    
    Args:
        text: Raw model response text
        
    Returns:
        Cleaned text with thinking blocks removed
    """
    if not text:
        return ""
    
    # Remove thinking/reasoning blocks (common in thinking models)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text).strip()
    
    return text


def normalize_url(url: str) -> str:
    """Normalize URL for comparison by removing protocol and trailing slashes.
    
    Args:
        url: URL string to normalize
        
    Returns:
        Normalized URL in lowercase (netloc + path)
    """
    if not url:
        return ""
    
    url = url.strip().rstrip("/")
    parsed = urlparse(url)
    normalized = f"{parsed.netloc}{parsed.path}".rstrip("/")
    return normalized.lower()


def load_judge_prompt(prompt_type: str = "baseline") -> str:
    """Load the judge prompt template.
    
    Args:
        prompt_type: Type of prompt ("baseline" or "rag")
        
    Returns:
        Prompt template string
        
    Raises:
        FileNotFoundError: If prompt file doesn't exist
    """
    prompt_filename = f"judge_prompt_{prompt_type}.txt"
    prompt_path = Path(__file__).parent.parent / "prompts" / prompt_filename
    
    if not prompt_path.exists():
        # Fallback to strict if specific prompt doesn't exist
        prompt_path = Path(__file__).parent.parent / "prompts" / "judge_prompt_strict.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Judge prompt not found: {prompt_path}")
    
    return prompt_path.read_text(encoding="utf-8")


def setup_judge_client(judge_model: str) -> openai.Client:
    """Setup OpenAI client for judge API.
    
    Args:
        judge_model: Judge model identifier
        
    Returns:
        Configured OpenAI client
        
    Raises:
        ValueError: If required API keys are missing
    """
    is_cscs_judge = "/" in judge_model or judge_model not in ["gpt-4o", "gpt-4-turbo"]
    
    if is_cscs_judge:
        api_key = os.getenv("CSCS_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("JUDGE_API_KEY")
        base_url = os.getenv("CSCS_BASE_URL") or os.getenv("JUDGE_BASE_URL") or "https://api.swissai.cscs.ch/v1"
        
        if not api_key:
            raise ValueError("CSCS_API_KEY, LLM_API_KEY, or JUDGE_API_KEY environment variable required")
        
        return openai.Client(api_key=api_key, base_url=base_url)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required")
        
        return openai.Client(api_key=api_key)


def extract_judge_scores(response_text: str) -> Dict[str, Any]:
    """Extract scores from judge response using multiple strategies.
    
    Args:
        response_text: Raw response text from judge model
        
    Returns:
        Dictionary with correctness, completeness, result_tag, reasoning
    """
    scores = None
    
    # Strategy 1: Look for JSON in code blocks
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if json_match:
        try:
            scores = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Strategy 2: Look for JSON object in the text
    if scores is None:
        json_match = re.search(r"\{[^{}]*\"correctness\"[^{}]*\"completeness\"[^{}]*\"result_tag\"[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            try:
                scores = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
    
    # Strategy 3: Extract individual fields with regex
    if scores is None:
        correctness_match = None
        completeness_match = None
        result_tag = None
        
        # Extract correctness (0-2 integer scale)
        patterns_correctness = [
            r"1\.\s*FACTUAL\s+CORRECTNESS.*?[Ss]core[:\s]+(\d)",
            r"FACTUAL\s+CORRECTNESS.*?[Ss]core[:\s]+(\d)",
            r"FACTUAL\s+CORRECTNESS.*?[Ss]core\s+(?:must\s+be|is)[:\s]+(\d)",
            r'"correctness"\s*:\s*(\d)',
            r'correctness["\']?\s*[:=]\s*(\d)',
            r"(?:correctness|CORRECTNESS|Factual\s+Correctness).*?[Ss]core[:\s]+(\d)",
        ]
        
        for pattern in patterns_correctness:
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    val = int(match.group(1))
                    if 0 <= val <= 2:
                        correctness_match = val
                        break
                except (ValueError, IndexError):
                    continue
        
        # Extract completeness (0-2 integer scale)
        patterns_completeness = [
            r"2\.\s*COMPLETENESS.*?[Ss]core[:\s]+(\d)",
            r"COMPLETENESS.*?[Ss]core[:\s]+(\d)",
            r"COMPLETENESS.*?[Ss]core\s+(?:must\s+be|is)[:\s]+(\d)",
            r'"completeness"\s*:\s*(\d)',
            r'completeness["\']?\s*[:=]\s*(\d)',
            r"(?:completeness|COMPLETENESS|Completeness).*?[Ss]core[:\s]+(\d)",
        ]
        
        for pattern in patterns_completeness:
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    val = int(match.group(1))
                    if 0 <= val <= 2:
                        completeness_match = val
                        break
                except (ValueError, IndexError):
                    continue
        
        # Extract result_tag
        tag_patterns = [
            r'"result_tag"\s*:\s*["\']?(Correct|Generic|Refusal|Hallucination|Partial)["\']?',
            r"result_tag[:\s]+[\"']?(Correct|Generic|Refusal|Hallucination|Partial)[\"']?",
            r"tag[:\s]+[\"']?(Correct|Generic|Refusal|Hallucination|Partial)[\"']?",
            r'result_tag["\']?\s*[:=]\s*["\']?(Correct|Generic|Refusal|Hallucination|Partial)["\']?',
        ]
        
        for pattern in tag_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                result_tag = match.group(1).capitalize()
                break
        
        # Infer tag from reasoning if not found
        if result_tag is None:
            text_lower = response_text.lower()
            if "hallucination" in text_lower or "invents" in text_lower or "invented" in text_lower:
                result_tag = "Hallucination"
            elif any(phrase in text_lower for phrase in ["refusal", "don't know", "cannot answer", "i don't know", "i cannot"]):
                result_tag = "Refusal"
            elif correctness_match == 2 and completeness_match == 2:
                result_tag = "Correct"
            elif "generic" in text_lower or "general" in text_lower or "general advice" in text_lower:
                result_tag = "Generic"
            else:
                result_tag = "Generic"
        
        # Create scores dict from extracted values
        if correctness_match is not None and completeness_match is not None:
            scores = {
                "correctness": correctness_match,
                "completeness": completeness_match,
                "result_tag": result_tag,
                "reasoning": response_text,
            }
        elif correctness_match is not None:
            completeness_match = 0 if correctness_match == 0 else max(0, correctness_match - 1)
            scores = {
                "correctness": correctness_match,
                "completeness": completeness_match,
                "result_tag": result_tag if result_tag else "Generic",
                "reasoning": response_text,
            }
        elif completeness_match is not None:
            correctness_match = 0 if completeness_match == 0 else max(0, completeness_match - 1)
            scores = {
                "correctness": correctness_match,
                "completeness": completeness_match,
                "result_tag": result_tag if result_tag else "Generic",
                "reasoning": response_text,
            }
    
    # Strategy 4: Try to parse entire response as JSON
    if scores is None:
        try:
            scores = json.loads(response_text)
        except json.JSONDecodeError:
            pass
    
    # Last resort: infer from text analysis
    if scores is None:
        text_lower = response_text.lower()
        eth_tools = ['ethis', 'asvz', 'moodle', 'card-ethz', 'web print', 'staffnet', 'euler']
        generic_phrases = ['generic', 'general advice', 'does not mention', 'misses', 'completely generic']
        refusal_phrases = ["don't know", "cannot", "i don't", "refusal"]
        hallucination_phrases = ['invents', 'invented', 'hallucination', 'fake', 'does not exist']
        
        mentions_eth_tool = any(tool in text_lower for tool in eth_tools)
        is_generic = any(phrase in text_lower for phrase in generic_phrases)
        is_refusal = any(phrase in text_lower for phrase in refusal_phrases)
        is_hallucination = any(phrase in text_lower for phrase in hallucination_phrases)
        
        if mentions_eth_tool and not is_generic:
            correctness, completeness, result_tag = 2, 2, "Correct"
        elif is_refusal:
            correctness, completeness, result_tag = 0, 0, "Refusal"
        elif is_hallucination:
            correctness, completeness, result_tag = 0, 0, "Hallucination"
        else:
            correctness, completeness, result_tag = 0, 0, "Generic"
        
        scores = {
            "correctness": correctness,
            "completeness": completeness,
            "result_tag": result_tag,
            "reasoning": response_text,
        }
    
    # Fallback if still no scores
    if scores is None:
        scores = {
            "correctness": 0,
            "completeness": 0,
            "result_tag": "Generic",
            "reasoning": response_text,
        }
    
    # Validate and normalize scores (0-2 integer scale)
    correctness = max(0, min(2, int(scores.get("correctness", 0))))
    completeness = max(0, min(2, int(scores.get("completeness", 0))))
    result_tag = str(scores.get("result_tag", "Generic"))
    reasoning = str(scores.get("reasoning", ""))
    
    # Enforce tag-based scoring rules
    if result_tag == "Partial":
        correctness = max(1, correctness) if correctness > 0 else correctness
        completeness = max(1, completeness) if completeness > 0 else completeness
    elif result_tag in ["Generic", "Refusal"]:
        correctness = 0
        completeness = 0
    
    # Validate result_tag
    valid_tags = ["Correct", "Partial", "Generic", "Refusal", "Hallucination"]
    if result_tag not in valid_tags:
        result_tag = "Generic"
    
    # Enforce strict scoring: Generic/Refusal must be 0/0
    if result_tag in ["Generic", "Refusal"]:
        correctness = 0
        completeness = 0
    
    return {
        "correctness": correctness,
        "completeness": completeness,
        "result_tag": result_tag,
        "reasoning": reasoning,
    }


def call_judge(
    client: openai.Client,
    judge_model: str,
    question: str,
    ground_truth: str,
    model_response: str,
    prompt_template: str,
) -> Dict[str, Any]:
    """Call the judge LLM to score a response.
    
    Args:
        client: OpenAI client for judge API
        judge_model: Judge model identifier
        question: Original question
        ground_truth: Ground truth answer
        model_response: Model's response to score
        prompt_template: Prompt template string
        
    Returns:
        Dictionary with correctness, completeness, result_tag, reasoning, judge_model
        
    Raises:
        RuntimeError: If judge API call fails
    """
    prompt = prompt_template.format(
        question=question,
        ground_truth=ground_truth,
        model_response=model_response,
    )
    
    messages = [
        {
            "role": "system",
            "content": "You are a Strict Auditor for ETH Zurich. Respond only with valid JSON, no additional text.",
        },
        {"role": "user", "content": prompt},
    ]
    
    try:
        resp = client.chat.completions.create(
            model=judge_model,
            messages=messages,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=JUDGE_MAX_TOKENS,
        )
        
        # Handle thinking models (like Kimi-K2-Thinking) that use reasoning_content
        message = resp.choices[0].message
        response_text = None
        
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            response_text = message.reasoning_content.strip()
        elif hasattr(message, 'content') and message.content:
            response_text = message.content.strip()
        
        if not response_text:
            response_text = str(message) if message else ""
            if not response_text:
                raise ValueError("No content or reasoning_content in judge response. Message object: " + str(type(message)))
        
        # Extract scores using shared function
        scores = extract_judge_scores(response_text)
        scores["judge_model"] = judge_model
        
        return scores
        
    except Exception as e:
        import sys
        import traceback
        print(f"Error calling judge: {e}", file=sys.stderr)
        traceback.print_exc()
        raise RuntimeError(f"Judge API call failed: {str(e)}")


def load_responses_with_cleaning(json_path: Path) -> List[Dict[str, Any]]:
    """Load responses from JSON file and clean thinking blocks.
    
    Args:
        json_path: Path to responses JSON file
        
    Returns:
        List of response dictionaries with cleaned model_response text
    """
    with open(json_path, "r", encoding="utf-8") as f:
        responses = json.load(f)
    
    # Clean thinking blocks from all responses
    for response in responses:
        model_response = response.get("model_response", "")
        if model_response:
            response["model_response"] = clean_thinking_blocks(model_response)
    
    return responses


def save_scores_progress(output_file: Path, existing_scores: Dict[int, Dict], new_scores: List[Dict], current_index: int, total: int) -> None:
    """Save scoring progress to file.
    
    Args:
        output_file: Path to output scores file
        existing_scores: Dictionary of existing scores by question_id
        new_scores: List of newly scored entries
        current_index: Current question index (1-based)
        total: Total number of questions
    """
    all_scores = existing_scores.copy()
    for score in new_scores:
        all_scores[score["question_id"]] = score
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(list(all_scores.values()), f, indent=2, ensure_ascii=False)
    
    print(f"\nProgress saved ({current_index}/{total} questions)")

