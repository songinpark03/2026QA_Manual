import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import json
from datetime import datetime

# 압축해제
import zipfile, os

if not os.path.exists("final_patent_chunking_results.json"):
    with zipfile.ZipFile("data.zip", "r") as z:
        z.extractall(".")


# OpenAI API 설정
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class PatentQAChatbot:
    def __init__(self, json_file_path: str):
        """특허 QA 챗봇 초기화 (다중 문서 참조)"""
        print("🤖 특허 QA 챗봇을 초기화하는 중...")
        
        # JSON 파일 읽기
        self.patents_data = self._load_json(json_file_path)
        
        # 출원번호 리스트
        self.patent_ids = list(self.patents_data.keys())
        print(f"✓ 총 {len(self.patent_ids)}개 특허 문서 로드 완료")
        
        # 벡터화용 요약 텍스트 수집
        self.summaries = []
        for patent_id in self.patent_ids:
            summary = self.patents_data[patent_id].get('patent_summary', '')
            self.summaries.append(summary)
        
        # TF-IDF 벡터화
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=1
        )
        self.summary_vectors = self.vectorizer.fit_transform(self.summaries)
        
        print("✅ 챗봇 준비 완료!\n")
    
    def _load_json(self, json_file_path: str) -> dict:
        """JSON 파일 로드"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✓ JSON 파일 로드 완료")
            return data
        except FileNotFoundError:
            raise Exception(f"JSON 파일을 찾을 수 없습니다: {json_file_path}")
        except json.JSONDecodeError:
            raise Exception(f"JSON 파일 형식 오류: {json_file_path}")
    
    def _find_top_relevant_patents(self, question: str, top_k: int = 3) -> list:
        """
        질문과 가장 관련성 높은 특허 top_k개 찾기
        
        Args:
            question: 사용자 질문
            top_k: 상위 k개 특허 (최대 3개)
        
        Returns:
            [(patent_id, similarity_score, index), ...] 리스트
        """
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.summary_vectors).flatten()
        
        # 유사도가 0보다 큰 것만 필터링
        valid_indices = np.where(similarities > 0)[0]
        if len(valid_indices) == 0:
            return []
        
        # 상위 top_k개 인덱스 추출
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 유사도 0 초과만
                patent_id = self.patent_ids[idx]
                results.append((patent_id, similarities[idx], idx))
        
        return results
    
    def _get_content_chunks(self, patent_id: str) -> list:
        """특허의 content_chunks 가져오기"""
        patent_data = self.patents_data.get(patent_id, {})
        content_chunks = patent_data.get('content_chunks', [])
        
        # 각 청크의 텍스트만 추출
        chunk_texts = []
        for chunk in content_chunks:
            text = chunk.get('text', '')
            if text and text.strip():
                chunk_texts.append(text)
        
        return chunk_texts
    
    def _generate_answer_from_chunk(self, question: str, chunk: str) -> tuple:
        """청크에서 답변 생성"""
        prompt = f"""당신은 특허 전문가입니다. 다음 문서 내용을 바탕으로 질문에 답변해주세요.
문서에 없는 내용은 추측하지 말고, 문서에 명시된 내용만을 사용하세요.

문서 내용:
{chunk}

질문: {question}

답변:"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "정확한 정보만 제공하는 특허 분석 전문가"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            
            # 유효한 답변인지 확인
            has_answer = not any(phrase in answer.lower() for phrase in [
                "찾을 수 없습니다", "정보가 없습니다", "언급되지 않습니다",
                "나와 있지 않습니다", "확인할 수 없습니다"
            ])
            
            return answer, has_answer
            
        except Exception as e:
            return f"오류 발생: {e}", False
    
    def _get_answers_from_patent(self, question: str, patent_id: str) -> list:
        """
        하나의 특허 문서에서 모든 청크를 검토하여 유효한 답변 수집
        
        Returns:
            유효한 답변 리스트
        """
        chunks = self._get_content_chunks(patent_id)
        valid_answers = []
        
        for chunk in chunks:
            answer, has_answer = self._generate_answer_from_chunk(question, chunk)
            if has_answer:
                valid_answers.append(answer)
        
        return valid_answers
    
    def _synthesize_multi_patent_answers(self, question: str, patent_answers: dict) -> str:
        """
        여러 특허 문서의 답변들을 자연스럽게 종합
        
        Args:
            question: 질문
            patent_answers: {patent_id: [답변1, 답변2, ...], ...}
        
        Returns:
            종합된 최종 답변
        """
        if not patent_answers:
            return "해당 질문에 대한 정보를 찾을 수 없습니다."
        
        # 모든 답변을 하나로 합치기 (특허 구분 없이)
        all_answers = []
        for patent_id, answers in patent_answers.items():
            all_answers.extend(answers)
        
        if not all_answers:
            return "해당 질문에 대한 정보를 찾을 수 없습니다."
        
        # 모든 답변을 하나의 텍스트로
        combined_content = "\n\n".join(all_answers)
        
        synthesis_prompt = f"""당신은 특허 전문가입니다. 다음은 여러 특허 문서에서 추출한 정보들입니다.
이 정보들을 바탕으로 질문에 대해 자연스럽고 유기적인 하나의 답변을 작성해주세요.

중요: 
- "첫 번째 특허에서는...", "다른 특허에서는..." 같은 구분 표현을 사용하지 마세요
- 마치 하나의 완전한 문서를 읽고 답변하는 것처럼 자연스럽게 작성하세요
- 여러 출처의 내용을 매끄럽게 통합하여 전문가 답변으로 제시하세요
- 반복되는 내용은 한 번만 언급하고, 상충되는 정보가 있다면 통합적으로 설명하세요
- 출처나 출원번호 정보는 포함하지 마세요

질문: {question}

참고 정보:
{combined_content}

답변 (자연스럽고 통합된 하나의 답변):"""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "여러 출처의 정보를 자연스럽게 통합하여 하나의 완결된 전문가 답변을 제공하는 특허 분석 전문가"},
                    {"role": "user", "content": synthesis_prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            synthesized_answer = response.choices[0].message.content.strip()
            
            # 각주 제거 - 답변만 반환
            return synthesized_answer
            
        except Exception as e:
            return f"답변 종합 중 오류: {e}"
    
    def ask(self, question: str, verbose: bool = True, max_patents: int = 3) -> dict:
        """
        질문에 답변하기 (다중 문서 참조)
        
        Args:
            question: 사용자 질문
            verbose: 상세 정보 출력 여부
            max_patents: 참조할 최대 특허 문서 수 (기본값: 3)
        
        Returns:
            답변 정보를 담은 딕셔너리
        """
        if verbose:
            print(f"\n💬 질문: {question}")
            print("=" * 60)
        
        # 1. 관련 특허 top 3 찾기
        top_patents = self._find_top_relevant_patents(question, top_k=max_patents)
        
        if not top_patents:
            result = {
                "question": question,
                "answer": "관련 특허 문서를 찾을 수 없습니다.",
                "application_numbers": [],
                "similarity_scores": [],
                "timestamp": datetime.now().isoformat()
            }
            if verbose:
                print("❌ 관련 특허 문서를 찾을 수 없습니다.\n")
            return result
        
        if verbose:
            print(f"🔍 상위 {len(top_patents)}개 관련 특허 발견:")
            for i, (patent_id, sim, _) in enumerate(top_patents, 1):
                print(f"   {i}. 📋 출원번호: {patent_id} (유사도: {sim:.3f})")
        
        # 2. 각 특허에서 답변 수집
        patent_answers = {}
        total_chunks = 0
        total_valid = 0
        
        for patent_id, similarity, idx in top_patents:
            if verbose:
                print(f"\n📄 [{patent_id}] 분석 중...")
            
            answers = self._get_answers_from_patent(question, patent_id)
            chunks = self._get_content_chunks(patent_id)
            
            total_chunks += len(chunks)
            
            if answers:
                patent_answers[patent_id] = answers
                total_valid += len(answers)
                if verbose:
                    print(f"   ✓ {len(chunks)}개 청크 중 {len(answers)}개에서 답변 발견")
            else:
                if verbose:
                    print(f"   - {len(chunks)}개 청크 검토 완료 (유효 답변 없음)")
        
        if verbose:
            print(f"\n📊 총 {total_chunks}개 청크 검토, {total_valid}개 유효 답변 발견")
            print("🔍 답변 종합 중...")
        
        # 3. 최종 답변 종합
        final_answer = self._synthesize_multi_patent_answers(question, patent_answers)
        
        result = {
            "question": question,
            "answer": final_answer,
            "application_numbers": [p[0] for p in top_patents],
            "similarity_scores": [float(p[1]) for p in top_patents],
            "patents_with_answers": list(patent_answers.keys()),
            "total_chunks_reviewed": total_chunks,
            "total_valid_answers": total_valid,
            "timestamp": datetime.now().isoformat()
        }
        
        if verbose:
            print("\n📝 최종 답변:")
            print("-" * 60)
            print(final_answer)
            print("=" * 60 + "\n")
        
        return result
    
    def chat(self):
        """대화형 모드 시작"""
        print("="*60)
        print("🤖 특허 QA 챗봇 (대화형 모드 - 다중 문서 참조)")
        print("="*60)
        print("질문을 입력하세요. 종료하려면 'quit', 'exit', '종료' 입력")
        print("-"*60 + "\n")
        
        chat_history = []
        
        while True:
            try:
                question = input("💬 질문: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', '종료', 'q']:
                    print("\n👋 챗봇을 종료합니다. 감사합니다!")
                    break
                
                # 답변 생성 (최대 3개 특허 참조)
                result = self.ask(question, verbose=True, max_patents=3)
                
                # 히스토리 저장
                chat_history.append(result)
                
            except KeyboardInterrupt:
                print("\n\n👋 챗봇을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}\n")
        
        # 대화 히스토리 저장
        if chat_history:
            self.save_chat_history(chat_history)
    
    def save_chat_history(self, history: list, filename: str = "chat_history.json"):
        """대화 히스토리 저장"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            print(f"\n💾 대화 내역이 '{filename}'에 저장되었습니다.")
        except Exception as e:
            print(f"\n❌ 저장 실패: {e}")
    
    def batch_process(self, questions: list, output_file: str = "batch_results.json", max_patents: int = 3):
        """여러 질문을 배치로 처리 (다중 문서 참조)"""
        print(f"\n📦 배치 처리 시작: {len(questions)}개 질문 (최대 {max_patents}개 특허 참조)")
        print("="*60)
        
        results = []
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] 처리 중: {question[:50]}...")
            result = self.ask(question, verbose=False, max_patents=max_patents)
            results.append(result)
            print(f"✓ 완료 - {len(result['patents_with_answers'])}개 특허에서 답변 생성")
        
        # 결과 저장
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n✅ 배치 처리 완료! 결과가 '{output_file}'에 저장되었습니다.")
        except Exception as e:
            print(f"\n❌ 결과 저장 실패: {e}")
        
        return results
