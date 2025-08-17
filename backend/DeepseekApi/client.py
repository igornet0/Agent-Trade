from core import ds_settings
import requests
import json
import re

class DSClient:
    api_key = ds_settings.ds_api
    API_URL = "https://api.deepseek.com/v1/chat/completions"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def analyze_news(self, news_text: str) -> dict[str, int]:
        system_prompt = """
            Ты финансовый аналитик крипторынка. Проанализируй новость и оцени ее влияние на криптовалюты.
            Даже если новость не упоминает криптовалюты напрямую, определи возможное косвенное влияние через:
            - Геополитическую напряженность
            - Инфляционные риски
            - Изменения на энергетических рынках
            - Глобальную экономическую нестабильность
            
            Верни JSON-объект формата: {"BTC": число, "ETH": число, ...} 
            где число от -100 (крайне негативно) до 100 (крайне позитивно).
            Если влияние незначительно, используй значения близкие к 0.
        """
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": news_text}
            ],
            "temperature": 0.1,
            "max_tokens": 100,
            "stop": None
        }

        try:
            # Отправка запроса
            response = self.session.post(
                self.API_URL,
                json=payload,
                timeout=15
            )
            # response = requests.post(self.API_URL, json=payload)
            if response.status_code != 200:
                error_msg = f"API Error {response.status_code}: {response.text}"
                raise ConnectionError(error_msg)

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Network error: {str(e)}") from e
        
        try:
            # Извлечение JSON из ответа
            print(response.json())
            content = response.json()['choices'][0]['message']['content']
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if not json_match:
                raise ValueError("Не найден JSON в ответе")
                
            result = json.loads(json_match.group())
            
            # Валидация результата
            for ticker, score in result.items():
                if not isinstance(score, (int, float)):
                    raise TypeError(f"Некорректный тип значения для {ticker}")
                if score < -100 or score > 100:
                    raise ValueError(f"Значение {score} выходит за пределы диапазона")
            
            return result
            
        except Exception as e:
            print(f"Ошибка: {e}")
            return content
    

    def _parse_response(self, data: dict) -> dict[str, int]:
        """Парсит ответ API в формате: 
        {
            "analysis": [
                {"currency": "BTC", "impact_score": 85},
                {"currency": "ETH", "impact_score": -30}
            ]
        }
        """
        print(data)
        results = {}
        analysis = data.get("analysis", [])
        
        for item in analysis:
            currency = item.get("currency", "").upper()
            score = item.get("impact_score", 0)
            
            # Корректируем выход за диапазон
            if not (-100 <= score <= 100):
                score = max(min(score, 100), -100)
            
            if currency:
                results[currency] = int(score)
                
        return results