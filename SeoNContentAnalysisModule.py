import pandas as pd
from typing import List, Dict, Any
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import requests
from bs4 import BeautifulSoup
import re
from konlpy.tag import Mecab
import math
from collections import Counter

@dataclass
class BlogTemplate:
    name: str
    structure: List[str]
    seo_guidelines: Dict[str, Any]
    example: str

class KeywordExtractor:
    def __init__(self):
        self.mecab = Mecab()
        # 한국어 불용어 사전 확장
        self.stopwords = {
            '있다', '하다', '이다', '되다', '않다', '같다', '없다', '위하다', '받다', '통하다',
            '그', '이', '저', '것', '수', '등', '들', '및', '에서', '그리고', '또는', '및',
            '제', '더', '각', '왜', '몇', '또', '이런', '저런', '무슨', '어떤', '같은',
            '때문', '이상', '이하', '이전', '이후', '때', '중', '듯', '만큼', '정도',
            '를', '을', '는', '은', '이', '가', '에', '에서', '로', '으로', '와', '과',
            '아', '어', '나', '도', '만', '까지', '부터'
        }
        
    def preprocess_text(self, text: str) -> str:
        """텍스트 전처리 개선"""
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        # 이모지 제거
        text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
        # URL 제거
        text = re.sub(r'http\S+|www.\S+', '', text)
        # 특수문자 제거 (일부 한글 문장 부호 유지)
        text = re.sub(r'[^\w\s.,!?()\"\'~％％°℃·\u3131-\u3163\uac00-\ud7a3]', ' ', text)
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_nouns_and_phrases(self, text: str) -> List[str]:
        """명사와 의미있는 구문 추출 개선"""
        # 형태소 분석
        morphs = self.mecab.pos(text)
        
        extracted_terms = []
        temp_phrase = []
        
        for i, (word, pos) in enumerate(morphs):
            # 명사류 추출 (복합명사 처리 개선)
            if pos.startswith('N') and len(word) > 1:
                if word not in self.stopwords:
                    # 연속된 명사 확인
                    if i > 0 and morphs[i-1][1].startswith('N'):
                        compound_noun = morphs[i-1][0] + word
                        if compound_noun not in extracted_terms:
                            extracted_terms.append(compound_noun)
                    extracted_terms.append(word)
                    
            # 명사구 생성
            if pos.startswith('N') or pos.startswith('SN'):
                temp_phrase.append(word)
            else:
                if len(temp_phrase) > 1:
                    phrase = ' '.join(temp_phrase)
                    if len(phrase) > 3 and phrase not in extracted_terms:
                        extracted_terms.append(phrase)
                temp_phrase = []
        
        return list(set(extracted_terms))  # 중복 제거

    def calculate_tfidf(self, documents: List[str]) -> Dict[str, float]:
        """TF-IDF 점수 계산 개선"""
        # 문서를 문장 단위로 분리 (개선된 분리 로직)
        sentences = []
        for doc in documents:
            # 문장 분리 패턴 개선
            doc_sentences = re.split(r'(?<=[.!?])\s+', doc)
            sentences.extend([s.strip() for s in doc_sentences if s.strip()])
        
        # 각 문장에서 용어 추출
        term_freq = []
        doc_freq = Counter()
        
        for sentence in sentences:
            terms = self.extract_nouns_and_phrases(sentence)
            term_counts = Counter(terms)
            term_freq.append(term_counts)
            doc_freq.update(set(terms))
        
        # TF-IDF 계산 (개선된 가중치 적용)
        N = len(sentences)
        tfidf_scores = {}
        
        for idx, term_counts in enumerate(term_freq):
            for term, tf in term_counts.items():
                df = doc_freq[term]
                # 수정된 IDF 계산
                idf = math.log(N / (1 + df)) + 1
                
                if term not in tfidf_scores:
                    tfidf_scores[term] = 0
                # 문장 위치 가중치 적용
                position_weight = 1.0
                if idx < len(term_freq) * 0.2:  # 첫 20% 문장
                    position_weight = 1.2
                elif idx > len(term_freq) * 0.8:  # 마지막 20% 문장
                    position_weight = 1.1
                    
                tfidf_scores[term] += tf * idf * position_weight
        
        return tfidf_scores

    def extract_keywords(self, content: str, main_keyword: str, num_keywords: int = 5) -> List[str]:
        """주요 키워드 추출 개선"""
        # 전처리
        processed_text = self.preprocess_text(content)
        
        # 문단 단위로 분리 (개선된 분리 로직)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', processed_text) if p.strip()]
        
        # TF-IDF 점수 계산
        tfidf_scores = self.calculate_tfidf(paragraphs)
        
        # 메인 키워드 관련 용어 가중치 부여 (개선된 로직)
        for term, score in tfidf_scores.items():
            # 메인 키워드와의 관련성 검사
            if main_keyword.lower() in term.lower():
                tfidf_scores[term] *= 1.5
            elif any(k in term.lower() for k in main_keyword.lower().split()):
                tfidf_scores[term] *= 1.3
        
        # 상위 키워드 선택
        sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 중복 제거 및 필터링 (개선된 로직)
        final_keywords = []
        used_terms = set()
        
        for keyword, score in sorted_keywords:
            if len(final_keywords) >= num_keywords:
                break
                
            # 이미 포함된 단어가 있는지 확인
            should_skip = False
            for used_term in used_terms:
                if keyword in used_term or used_term in keyword:
                    should_skip = True
                    break
            
            # 키워드 품질 검사
            if not should_skip and keyword not in self.stopwords:
                if len(keyword) >= 2 and not keyword.isdigit():
                    final_keywords.append(keyword)
                    used_terms.add(keyword)
        
        return final_keywords


class SEOContentAnalyzer:
    def __init__(self):
        self.keyword_extractor = KeywordExtractor()
        self.templates = {
            "리뷰형": BlogTemplate(
                name="리뷰형",
                structure=[
                    "제품/서비스 소개",
                    "주요 특징 및 장점",
                    "사용 경험",
                    "비교 분석",
                    "최종 평가"
                ],
                seo_guidelines={
                    "제목 길이": "20-40자",
                    "문단 길이": "200-300자",
                    "키워드 밀도": "1-2%",
                    "이미지 수": "3-5개",
                    "내부 링크": "2-3개"
                },
                example="[솔직 리뷰] OO 제품 사용 후기 - 장단점 분석"
            ),
            "하우투형": BlogTemplate(
                name="하우투형",
                structure=[
                    "문제 정의",
                    "해결 방법 단계별 설명",
                    "팁과 주의사항",
                    "성공 사례",
                    "마무리"
                ],
                seo_guidelines={
                    "제목 길이": "30-50자",
                    "문단 길이": "150-250자",
                    "키워드 밀도": "2-3%",
                    "이미지 수": "4-6개",
                    "내부 링크": "3-4개"
                },
                example="OO 완벽 가이드 - 초보자도 쉽게 따라하는 방법"
            ),
            "정보형": BlogTemplate(
                name="정보형",
                structure=[
                    "주제 소개",
                    "배경 설명",
                    "주요 개념",
                    "상세 분석",
                    "결론"
                ],
                seo_guidelines={
                    "제목 길이": "25-45자",
                    "문단 길이": "250-350자",
                    "키워드 밀도": "1.5-2.5%",
                    "이미지 수": "2-4개",
                    "내부 링크": "4-5개"
                },
                example="OO 완벽 가이드 - 알아야 할 모든 것"
            )
        }
        
    def analyze_keyword_trends(self, keyword: str) -> Dict[str, Any]:
        """네이버 검색 트렌드 API를 활용한 키워드 트렌드 분석"""
        try:
            # 실제 구현시에는 네이버 검색 API 사용
            base_volume = np.random.randint(1000, 10000)
            monthly_trend = np.random.normal(loc=0, scale=5, size=12)  # 연간 추세
            seasonal_factor = np.sin(np.linspace(0, 2*np.pi, 12)) * 10  # 계절성
            
            current_month = datetime.now().month - 1
            current_volume = int(base_volume * (1 + monthly_trend[current_month]/100 + seasonal_factor[current_month]/100))
            prev_volume = int(base_volume * (1 + monthly_trend[current_month-1]/100 + seasonal_factor[current_month-1]/100))
            
            change_pct = ((current_volume - prev_volume) / prev_volume) * 100
            
            # 계절성 판단
            max_season_idx = np.argmax(seasonal_factor)
            seasons = ["봄", "여름", "가을", "겨울"]
            peak_season = seasons[max_season_idx // 3]
            
            trends = {
                "월간 검색량": current_volume,
                "전월 대비 증감": change_pct,
                "계절성": peak_season,
                "연관 키워드": self.keyword_extractor.extract_keywords(
                    f"{keyword} 관련 콘텐츠", keyword, num_keywords=5
                )
            }
            return trends
            
        except Exception as e:
            return {"error": str(e)}

    def _analyze_readability(self, content: str) -> Dict[str, Any]:
        """가독성 분석"""
        sentences = re.split(r'[.!?]\s+', content)
        words_per_sentence = [len(s.split()) for s in sentences if s.strip()]
        
        return {
            "평균 문장 길이": np.mean(words_per_sentence),
            "최장 문장 길이": max(words_per_sentence),
            "문장 길이 분포": {
                "짧은 문장(15단어 미만)": sum(1 for x in words_per_sentence if x < 15) / len(words_per_sentence),
                "중간 문장(15-25단어)": sum(1 for x in words_per_sentence if 15 <= x <= 25) / len(words_per_sentence),
                "긴 문장(25단어 초과)": sum(1 for x in words_per_sentence if x > 25) / len(words_per_sentence)
            }
        }

    def _generate_tags(self, content: str, main_keyword: str) -> List[str]:
        """컨텐츠 기반 태그 자동 생성 개선"""
        try:
            # 키워드 추출
            extracted_keywords = self.keyword_extractor.extract_keywords(content, main_keyword, num_keywords=8)
            
            # 태그 생성 및 정제
            tags = [main_keyword]  # 메인 키워드는 항상 첫 번째 태그
            
            for keyword in extracted_keywords:
                # 태그 정제
                tag = re.sub(r'\s+', '_', keyword.strip())  # 공백을 언더스코어로 변환
                tag = re.sub(r'[^\w\s가-힣]', '', tag)  # 특수문자 제거
                
                # 태그 유효성 검사
                if (tag and 
                    tag not in tags and 
                    tag != main_keyword and 
                    len(tag) >= 2 and 
                    not tag.isdigit()):
                    tags.append(tag)
            
            # 태그 정렬 및 제한
            tags = sorted(tags, key=len)[:10]  # 길이순 정렬 후 최대 10개로 제한
            
            return tags
            
        except Exception as e:
            print(f"태그 생성 중 오류 발생: {str(e)}")
            return [main_keyword]

    def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """컨텐츠 구조 분석 개선"""
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
        headers = re.findall(r'#{1,6}\s.+', content)
        images = len(re.findall(r'!\[.*?\]\(.*?\)', content))  # Markdown 이미지 검색
        
        # 문단 분석
        paragraph_lengths = [len(p) for p in paragraphs]
        
        structure_analysis = {
            "문단 수": len(paragraphs),
            "평균 문단 길이": np.mean(paragraph_lengths) if paragraph_lengths else 0,
            "최대 문단 길이": max(paragraph_lengths) if paragraph_lengths else 0,
            "제목 수": len(headers),
            "이미지 수": images,
            "문단 길이 분포": {
                "짧은 문단(200자 미만)": sum(1 for x in paragraph_lengths if x < 200),
                "적정 문단(200-500자)": sum(1 for x in paragraph_lengths if 200 <= x <= 500),
                "긴 문단(500자 초과)": sum(1 for x in paragraph_lengths if x > 500)
            }
        }
        
        # 개선 필요 사항 분석
        improvements = []
        
        if not headers:
            improvements.append("제목 구조가 없습니다. 주제별로 제목을 추가하여 가독성을 높이세요.")
        elif len(headers) < 3:
            improvements.append("제목 구조가 부족합니다. 더 많은 소제목을 추가하여 내용을 체계적으로 구성하세요.")
            
        if images < 2:
            improvements.append("이미지가 부족합니다. 시각적 요소를 추가하여 내용의 이해도를 높이세요.")
            
        if structure_analysis["문단 길이 분포"]["긴 문단(500자 초과)"] > 0:
            improvements.append("일부 문단이 너무 깁니다. 500자 이하로 나누어 가독성을 개선하세요.")
            
        if structure_analysis["평균 문단 길이"] < 100:
            improvements.append("문단이 너무 짧습니다. 내용을 더 자세히 설명하여 풍부한 정보를 제공하세요.")
            
        structure_analysis["개선 필요 사항"] = improvements
        return structure_analysis

    def calculate_seo_score(self, content: str, keyword: str) -> Dict[str, Any]:
        """SEO 점수 계산 개선"""
        # 기본 분석
        word_count = len(content.split())
        paragraph_analysis = self._analyze_content_structure(content)
        readability_analysis = self._analyze_readability(content)
        
        # 점수 구성 요소 계산
        score_components = {
            "길이 점수": min(100, word_count / 20),  # 최소 2000자 기준
            "키워드 밀도 점수": self._calculate_keyword_density_score(content, keyword),
            "구조 점수": self._calculate_structure_score(content, paragraph_analysis),
            "가독성 점수": self._calculate_readability_score(readability_analysis),
            "최적화 점수": self._calculate_optimization_score(content, keyword)
        }
        
        # 가중치 적용
        weights = {
            "길이 점수": 0.2,
            "키워드 밀도 점수": 0.25,
            "구조 점수": 0.2,
            "가독성 점수": 0.2,
            "최적화 점수": 0.15
        }
        
        weighted_score = sum(score * weights[key] for key, score in score_components.items())
        
        return {
            "총점": round(weighted_score, 2),
            "세부 점수": score_components,
            "등급": self._get_grade(weighted_score),
            "분석 결과": {
                "문단 분석": paragraph_analysis,
                "가독성 분석": readability_analysis
            }
        }

    def _calculate_keyword_density_score(self, content: str, keyword: str) -> float:
        """키워드 밀도 점수 계산 개선"""
        # 전처리
        processed_content = content.lower()
        processed_keyword = keyword.lower()
        
        # 키워드 변형 고려
        keyword_variants = [
            processed_keyword,
            processed_keyword.replace(' ', ''),
            processed_keyword.replace('-', ' '),
            processed_keyword.replace('_', ' ')
        ]
        
        # 총 단어 수
        total_words = len(processed_content.split())
        
        # 키워드 출현 횟수 (변형 포함)
        keyword_count = sum(processed_content.count(variant) for variant in keyword_variants)
        
        # 밀도 계산
        density = (keyword_count / total_words) * 100
        
        # 점수 계산 (1-3% 사이가 최적)
        if 1 <= density <= 3:
            return 100
        elif density < 1:
            return max(0, 100 - (1 - density) * 50)
        else:
            return max(0, 100 - (density - 3) * 25)

    def _calculate_structure_score(self, content: str, analysis: Dict) -> float:
        """구조 점수 계산"""
        score = 100
        
        # 제목 구조 평가
        if analysis["제목 수"] == 0:
            score -= 30
        elif analysis["제목 수"] < 3:
            score -= 15
        
        # 문단 길이 평가
        if analysis["문단 길이 분포"]["긴 문단(500자 초과)"] > 0:
            score -= 10 * analysis["문단 길이 분포"]["긴 문단(500자 초과)"]
        
        # 이미지 수 평가
        if analysis["이미지 수"] < 2:
            score -= 15
        
        return max(0, score)

    def _calculate_readability_score(self, analysis: Dict) -> float:
        """가독성 점수 계산"""
        score = 100
        
        # 평균 문장 길이 평가 (15-25 단어가 이상적)
        avg_sentence_length = analysis["평균 문장 길이"]
        if avg_sentence_length > 25:
            score -= min(30, (avg_sentence_length - 25) * 2)
        elif avg_sentence_length < 15:
            score -= min(30, (15 - avg_sentence_length) * 2)
        
        # 문장 길이 분포 평가
        distribution = analysis["문장 길이 분포"]
        if distribution["긴 문장(25단어 초과)"] > 0.2:
            score -= min(30, (distribution["긴 문장(25단어 초과)"] - 0.2) * 100)
        
        return max(0, score)

    def _calculate_optimization_score(self, content: str, keyword: str) -> float:
        """최적화 점수 계산"""
        score = 100
        
        # 제목에 키워드 포함 여부
        headers = re.findall(r'#{1,6}\s.+', content)
        if not any(keyword.lower() in h.lower() for h in headers):
            score -= 20
        
        # 첫 문단에 키워드 포함 여부
        paragraphs = content.split('\n\n')
        if paragraphs and keyword.lower() not in paragraphs[0].lower():
            score -= 15
        
        # 메타 설명 길이 (첫 문단 기준)
        if paragraphs and len(paragraphs[0]) < 100:
            score -= 10
        
        return max(0, score)

    def _get_grade(self, score: float) -> str:
        """점수를 등급으로 변환"""
        if score >= 90: return "S"
        elif score >= 80: return "A"
        elif score >= 70: return "B"
        elif score >= 60: return "C"
        else: return "D"