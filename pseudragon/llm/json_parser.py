"""
Helper function to safely parse JSON from LLM responses
LLM 응답에서 JSON을 안전하게 파싱하는 헬퍼 함수

Handles various response formats from different LLM providers.
다양한 LLM 제공자의 응답 형식을 처리합니다.
"""

# Standard library imports
# 표준 라이브러리 import
import json
import re
from typing import Any, Dict


def normalize_json_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize JSON keys to match expected format.
    LLM이 다양한 키 형식을 사용할 수 있으므로 표준 형식으로 정규화합니다.
    
    Handles variations like:
    - "Is PII" / "is_pii" / "isPII" → "is_pii"
    - "PII Type" / "pii_type" / "piiType" → "pii_type"
    - "Reason" / "reasoning" → "reasoning"
    
    Args:
        data: Parsed JSON dictionary
        
    Returns:
        Normalized JSON dictionary
    """
    normalized = {}

    # Key mapping rules (case-insensitive, space/underscore agnostic)
    # 키 매핑 규칙 (대소문자 무시, 공백/언더스코어 무관)
    key_mappings = {
        "is_pii": ["is_pii", "ispii", "is pii", "pii"],
        "pii_type": ["pii_type", "piitype", "pii type", "type"],
        "reasoning": ["reasoning", "reason", "rationale", "explanation"],
        "evidence_source": ["evidence_source", "evidencesource", "evidence source", "evidence", "source"],
        "chain_of_thought": ["chain_of_thought", "chainofthought", "chain of thought", "cot"],
        "recommended_methods": ["recommended_methods", "recommendedmethods", "recommended methods", "methods"],
    }

    # First, copy all original keys
    # 먼저 모든 원본 키를 복사
    for key, value in data.items():
        normalized[key] = value

    # Then, normalize known keys
    # 그 다음 알려진 키들을 정규화
    for standard_key, variations in key_mappings.items():
        for original_key in data.keys():
            # Normalize the original key for comparison (lowercase, remove spaces/underscores)
            # 비교를 위해 원본 키를 정규화 (소문자, 공백/언더스코어 제거)
            normalized_original = original_key.lower().replace(" ", "").replace("_", "")

            # Check if it matches any variation
            # 변형 중 하나와 일치하는지 확인
            for variation in variations:
                normalized_variation = variation.lower().replace(" ", "").replace("_", "")
                if normalized_original == normalized_variation:
                    # Map to standard key
                    # 표준 키로 매핑
                    normalized[standard_key] = data[original_key]
                    break

    return normalized


def safe_parse_json(content: str, default_response: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Safely parse JSON from LLM response content with key normalization.
    LLM 응답 콘텐츠에서 JSON을 안전하게 파싱하고 키를 정규화합니다.
    
    Args:
        content: Response content from LLM
        default_response: Default response if parsing fails
        
    Returns:
        Parsed and normalized JSON dictionary
    """
    if not content:
        print("[WARNING] [JSON Parser] Empty response from LLM")
        return default_response or {"is_pii": False, "pii_type": "Unknown", "reasoning": "Empty response from LLM"}

    parsed_json = None

    # Try direct JSON parsing first
    # 먼저 직접 JSON 파싱 시도
    try:
        parsed_json = json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in markdown (```json ... ```)
    # 마크다운 JSON 블록 찾기 시도 (```json ... ```)
    if not parsed_json:
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                parsed_json = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

    # Try to find any JSON object in the text
    # 텍스트에서 JSON 객체 찾기 시도
    if not parsed_json:
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
        if json_match:
            try:
                parsed_json = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

    # If parsing succeeded, normalize keys
    # 파싱 성공 시 키 정규화
    if parsed_json:
        # Check if it's an empty JSON object
        # 빈 JSON 객체인지 확인
        if not parsed_json or parsed_json == {}:
            print("[WARNING] [JSON Parser] LLM returned empty JSON object: {}")
            print(f"   Raw content preview: {content[:200]}")
            return default_response or {"is_pii": False, "pii_type": "Unknown", "reasoning": "LLM returned empty JSON"}

        # Normalize keys to standard format
        # 키를 표준 형식으로 정규화
        normalized = normalize_json_keys(parsed_json)

        # Validate required fields only if default_response expects them
        # Stage 1 expects "is_pii", Stage 2 expects "recommended_methods"
        # default_response가 기대하는 필드만 검증
        # Stage 1은 "is_pii" 기대, Stage 2는 "recommended_methods" 기대
        if default_response and "is_pii" in default_response and "is_pii" not in normalized:
            print(f"[WARNING] [JSON Parser] Missing 'is_pii' field in response")
            print(f"   Original keys: {list(parsed_json.keys())}")
            print(f"   Normalized keys: {list(normalized.keys())}")
            print(f"   Raw content preview: {content[:200]}")

            # Use default if provided
            # 기본값이 제공된 경우 사용
            return default_response

        return normalized

    # If all parsing attempts fail, return default or error
    # 모든 파싱 시도 실패 시 기본값 또는 오류 반환
    print(f"[WARNING] [JSON Parser] Failed to parse JSON from LLM response")
    print(f"   Raw content preview: {content}")

    if default_response:
        return default_response

    return {"is_pii": False, "pii_type": "Unknown", "reasoning": "Failed to parse LLM response", "error": "Failed to parse JSON", "raw_content": content}
