from django.views.generic import TemplateView
from web_project import TemplateLayout
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin
from apps.authentication.models import ViewHistory
import pickle
import re
import nltk
import os
import torch
import xgboost as xgb
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from PyPDF2 import PdfReader
import tempfile
import pytesseract
from pdf2image import convert_from_path

# Download required NLTK data
def download_nltk_data():
    try:
        # Download punkt (tokenizer)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("Downloading punkt tokenizer...")
            nltk.download('punkt', quiet=True)
            
        # Download punkt_tab
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print("Downloading punkt_tab...")
            nltk.download('punkt_tab', quiet=True)
            
        # Download stopwords
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            print("Downloading stopwords...")
            nltk.download('stopwords', quiet=True)
            
        # Download wordnet
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading wordnet...")
            nltk.download('wordnet', quiet=True)
            
        # Download averaged_perceptron_tagger (for better tokenization)
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            print("Downloading averaged_perceptron_tagger...")
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
        print("All NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"Error downloading NLTK data: {str(e)}")
        return False

# Download NLTK data when module is imported
nltk_data_loaded = download_nltk_data()

# Initialize models as None
bert_model = None
xgb_model = None
rf_model = None

def load_models():
    global bert_model, xgb_model, rf_model
    try:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_dir = os.path.join(BASE_DIR, 'models')
        
        # Create models directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print(f"Created models directory at: {model_dir}")
            return False

        # Load BERT model
        bert_path = os.path.join(model_dir, 'bert_model.pkl')
        if os.path.exists(bert_path):
            # Initialize BERT model directly instead of loading from pickle
            bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("BERT model initialized successfully")
        else:
            print(f"Using default BERT model")
            bert_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load XGBoost model
        xgb_path = os.path.join(model_dir, 'xgb_model.pkl')
        if os.path.exists(xgb_path):
            try:
                # Try loading with xgb.Booster first
                xgb_model = xgb.Booster()
                xgb_model.load_model(xgb_path)
                print("XGBoost model loaded successfully using Booster")
            except Exception as e:
                print(f"Error loading XGBoost model with Booster: {str(e)}")
                try:
                    # Fall back to pickle if Booster fails
                    with open(xgb_path, 'rb') as f:
                        xgb_model = pickle.load(f)
                    print("XGBoost model loaded successfully using pickle")
                except Exception as e:
                    print(f"Error loading XGBoost model with pickle: {str(e)}")
                    return False
        else:
            print(f"XGBoost model not found at: {xgb_path}")
            return False

        # Load Random Forest model
        rf_path = os.path.join(model_dir, 'rf_model.pkl')
        if os.path.exists(rf_path):
            with open(rf_path, 'rb') as f:
                rf_model = pickle.load(f)
            print("Random Forest model loaded successfully")
        else:
            print(f"Random Forest model not found at: {rf_path}")
            return False

        print("All models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

# Try to load models when the module is imported
models_loaded = load_models()

def preprocess_text(text_list):
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        preprocessed = []

        for sentence in text_list:
            # Remove URLs
            sentence = re.sub(r'http\S+', '', sentence)
            # Remove special characters and numbers
            sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
            # Convert to lowercase
            sentence = sentence.lower()
            # Tokenize
            words = word_tokenize(sentence)
            # Remove stopwords and lemmatize
            words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
            preprocessed.append(' '.join(words))

        return preprocessed
    except Exception as e:
        print(f"Error in preprocess_text: {str(e)}")
        raise

"""
This file is a view controller for multiple pages as a module.
Here you can override the page view layout.
Refer to dashboards/urls.py file for more pages.
"""


class DashboardsView(LoginRequiredMixin, TemplateView):
    login_url = '/login/'
    redirect_field_name = 'next'

    # Predefined function
    def get_context_data(self, **kwargs):
        # A function to init the global layout. It is defined in web_project/__init__.py file
        context = TemplateLayout.init(self, super().get_context_data(**kwargs))

        return context

def extract_text_with_ocr(pdf_path):
    try:
        # First try normal text extraction
        pdf_reader = PdfReader(pdf_path)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        
        # If no text was extracted, try OCR
        if not text.strip():
            print("No text extracted normally, attempting OCR...")
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            text = ""
            
            # Process each page
            for i, image in enumerate(images):
                print(f"Processing page {i+1} with OCR...")
                # Convert image to text using OCR
                page_text = pytesseract.image_to_string(image)
                text += page_text + "\n"
                
            if not text.strip():
                raise ValueError("No text could be extracted even with OCR")
                
        return text
    except Exception as e:
        print(f"Error in OCR processing: {str(e)}")
        raise

@login_required(login_url='/auth/login/')
@csrf_protect
def analyze_pdf(request):
    if request.method == 'POST':
        try:
            if 'file' not in request.FILES:
                return JsonResponse({
                    'status': 'error',
                    'error': 'No file uploaded'
                }, status=400)
                
            uploaded_file = request.FILES['file']
            
            # Log file details for debugging
            print(f"Received file: {uploaded_file.name}")
            print(f"Content type: {uploaded_file.content_type}")
            print(f"Size: {uploaded_file.size} bytes")
            
            # Validate file type
            valid_mime_types = ['application/pdf', 'application/x-pdf']
            content_type = uploaded_file.content_type.lower()
            print(f"File content type: {content_type}")
            
            if not any(mime_type in content_type for mime_type in valid_mime_types):
                return JsonResponse({
                    'status': 'error',
                    'error': f'Invalid file type: {content_type}. Please upload a PDF file.'
                }, status=415)
                
            # Validate file extension
            if not uploaded_file.name.lower().endswith('.pdf'):
                return JsonResponse({
                    'status': 'error',
                    'error': 'Invalid file extension. Please upload a file with .pdf extension.'
                }, status=415)
                
            # Validate file size (10MB limit)
            if uploaded_file.size > 10 * 1024 * 1024:
                return JsonResponse({
                    'status': 'error',
                    'error': f'File size ({uploaded_file.size / 1024 / 1024:.1f}MB) exceeds maximum limit of 10MB'
                }, status=413)
                
            try:
                # Create a temporary file to store the PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf', mode='wb') as temp_file:
                    try:
                        # Write file chunks
                        for chunk in uploaded_file.chunks():
                            temp_file.write(chunk)
                        temp_file.flush()
                        os.fsync(temp_file.fileno())
                        
                        print(f"Temporary file created: {temp_file.name}")
                        
                        try:
                            # Extract text using OCR if needed
                            text = extract_text_with_ocr(temp_file.name)
                            
                            if not text.strip():
                                raise ValueError("No text could be extracted from the PDF")
                            
                            # Preprocess and predict with existing models
                            processed_text = preprocess_text([text])
                            vectorized = bert_model.encode(processed_text)

                            # Convert to numpy array and ensure it's 2D
                            vectorized = np.array(vectorized)
                            if len(vectorized.shape) == 1:
                                vectorized = vectorized.reshape(1, -1)

                            # Get predictions from both models
                            xgb_pred = xgb_model.predict(vectorized)[0]
                            rf_pred = rf_model.predict(vectorized)[0]

                            # Get probabilities from both models
                            xgb_proba = xgb_model.predict_proba(vectorized)[0][1]
                            rf_proba = rf_model.predict_proba(vectorized)[0][1]

                            # Average the predictions
                            final_prediction = (xgb_pred + rf_pred) / 2
                            final_confidence = (xgb_proba + rf_proba) / 2

                            # Determine if it's fake or real
                            is_fake = final_prediction < 0.5
                            confidence_score = int(final_confidence * 100)

                            # Generate initial analysis
                            text_analysis = {
                                'accuracy': confidence_score,
                                'credibility': int((1 - final_confidence) * 100),
                                'sentiment': 'Negative' if is_fake else 'Positive',
                                'source_reliability': int((1 - final_confidence) * 100),
                                'is_fake': int(is_fake)
                            }

                            # Get Llama model analysis
                            llama_analysis = analyze_with_llama(text)

                            # Combine both analyses
                            final_analysis = combine_analysis_results(text_analysis, llama_analysis)

                            # Add file information
                            final_analysis['file_name'] = uploaded_file.name

                            # Create view history entry
                            ViewHistory.objects.create(
                                user=request.user,
                                content_type='file',
                                content=uploaded_file.name,
                                result=not is_fake,  # True for real news, False for fake news
                                confidence=final_confidence
                            )
                            
                            return JsonResponse({
                                'status': 'success',
                                'data': final_analysis,
                                'message': 'PDF analyzed successfully'
                            })
                            
                        except Exception as e:
                            print(f"Error processing PDF: {str(e)}")
                            return JsonResponse({
                                'status': 'error',
                                'error': f'Error processing PDF: {str(e)}'
                            }, status=422)
                    except Exception as e:
                        print(f"Error writing file chunks: {str(e)}")
                        return JsonResponse({
                            'status': 'error',
                            'error': f'Error saving uploaded file: {str(e)}'
                        }, status=500)
                    finally:
                        # Clean up the temporary file
                        try:
                            if os.path.exists(temp_file.name):
                                os.unlink(temp_file.name)
                                print("Temporary file cleaned up")
                        except Exception as e:
                            print(f"Error cleaning up temporary file: {str(e)}")
                
            except Exception as e:
                print(f"Error in file processing: {str(e)}")
                return JsonResponse({
                    'status': 'error',
                    'error': f'Error processing PDF: {str(e)}'
                }, status=500)
                
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'error': f'Unexpected error: {str(e)}'
            }, status=500)
    
    return JsonResponse({
        'status': 'error',
        'error': 'Invalid request method'
    }, status=405)

def analyze_with_llama(text):
    try:
        api_key = os.getenv('OPEN_ROUTER_API_KEY')
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        prompt = f"""Analyze the following news content and determine if it's likely to be fake or real news. 
        Consider factors like credibility, consistency, and potential biases. 
        Provide a detailed analysis in JSON format with the following structure:
        {{
            "is_fake": boolean,
            "confidence": float (0-1),
            "analysis": string,
            "key_points": array of strings
        }}

        News content: {text}"""

        try:
            completion = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": os.getenv('BASE_URL'),
                    "X-Title": "FakeGuard",
                    "Authorization": f"Bearer {api_key}" 
                },
                model="qwen/qwen3-30b-a3b:free",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Parse the response
            response = completion.choices[0].message.content
            # Extract JSON from the response
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {
                    "is_fake": False,
                    "confidence": 0.5,
                    "analysis": "Unable to parse model response",
                    "key_points": []
                }
        except Exception as e:
            error_message = str(e)
            if "No endpoints found matching your data policy" in error_message:
                # Provide a fallback analysis when OpenRouter is not configured
                return {
                    "is_fake": False,
                    "confidence": 0.5,
                    "analysis": "LLM analysis is currently unavailable. Please check OpenRouter configuration.",
                    "key_points": [
                        "LLM analysis requires proper OpenRouter configuration",
                        "Please enable prompt training in OpenRouter settings",
                        "Visit https://openrouter.ai/settings/privacy to configure"
                    ]
                }
            else:
                raise e
    except Exception as e:
        print(f"Error in open router request analysis: {str(e)}")
        return {
            "is_fake": False,
            "confidence": 0.5,
            "analysis": f"Error in analysis: {str(e)}",
            "key_points": [
                "An error occurred during LLM analysis",
                "Please check the server logs for details"
            ]
        }

def combine_analysis_results(text_analysis, llama_analysis):
    # If LLM analysis failed, rely more on the ML analysis
    if "Error in analysis" in llama_analysis['analysis'] or "LLM analysis is currently unavailable" in llama_analysis['analysis']:
        return {
            'is_fake': text_analysis['is_fake'],
            'accuracy': text_analysis['accuracy'],
            'credibility': text_analysis['credibility'],
            'sentiment': text_analysis['sentiment'],
            'source_reliability': text_analysis['source_reliability'],
            'analysis': "Analysis based on machine learning models only. LLM analysis is currently unavailable.",
            'key_points': [
                f"Machine Learning Analysis: {text_analysis['sentiment']} sentiment with {text_analysis['accuracy']}% confidence",
                f"Source Reliability: {text_analysis['source_reliability']}%",
                *llama_analysis['key_points']  # Include the error/configuration messages
            ]
        }
    
    # Normal combination when both analyses are available
    combined_confidence = (text_analysis['accuracy'] + (llama_analysis['confidence'] * 100)) / 2
    
    # Determine final verdict based on both analyses
    is_fake = text_analysis['is_fake'] or llama_analysis['is_fake']
    
    # Combine key points
    key_points = [
        f"Machine Learning Analysis: {text_analysis['sentiment']} sentiment with {text_analysis['accuracy']}% confidence",
        f"Source Reliability: {text_analysis['source_reliability']}%",
        f"LLM Analysis: {llama_analysis['analysis']}",
        *llama_analysis['key_points']
    ]
    
    return {
        'is_fake': int(is_fake),
        'accuracy': int(combined_confidence),
        'credibility': int(100 - combined_confidence),
        'sentiment': 'Negative' if is_fake else 'Positive',
        'source_reliability': text_analysis['source_reliability'],
        'analysis': llama_analysis['analysis'],
        'key_points': key_points
    }

@login_required(login_url='/login/')
@csrf_protect
def analyze_text(request):
    if request.method == 'POST':
        try:
            # Check if NLTK data is loaded
            if not nltk_data_loaded:
                return JsonResponse({
                    'status': 'error',
                    'error': 'NLTK data not loaded. Please check server logs for details.'
                }, status=500)

            # Check if models are loaded
            if not models_loaded:
                return JsonResponse({
                    'status': 'error',
                    'error': 'Models not loaded. Please check server logs for details.'
                }, status=500)

            text = request.POST.get('text', '').strip()
            if not text:
                return JsonResponse({
                    'status': 'error',
                    'error': 'No text provided'
                }, status=400)

            # Preprocess and predict with existing models
            processed_text = preprocess_text([text])
            vectorized = bert_model.encode(processed_text)

            # Convert to numpy array and ensure it's 2D
            vectorized = np.array(vectorized)
            if len(vectorized.shape) == 1:
                vectorized = vectorized.reshape(1, -1)

            # Get predictions from both models
            xgb_pred = xgb_model.predict(vectorized)[0]
            rf_pred = rf_model.predict(vectorized)[0]

            # Get probabilities from both models
            xgb_proba = xgb_model.predict_proba(vectorized)[0][1]
            rf_proba = rf_model.predict_proba(vectorized)[0][1]

            # Average the predictions
            final_prediction = (xgb_pred + rf_pred) / 2
            final_confidence = (xgb_proba + rf_proba) / 2

            # Determine if it's fake or real
            is_fake = final_prediction < 0.5
            confidence_score = int(final_confidence * 100)

            # Generate initial analysis
            text_analysis = {
                'accuracy': confidence_score,
                'credibility': int((1 - final_confidence) * 100),
                'sentiment': 'Negative' if is_fake else 'Positive',
                'source_reliability': int((1 - final_confidence) * 100),
                'is_fake': int(is_fake)
            }

            # Get Llama model analysis
            llama_analysis = analyze_with_llama(text)

            # Combine both analyses
            final_analysis = combine_analysis_results(text_analysis, llama_analysis)

            # Create view history entry
            ViewHistory.objects.create(
                user=request.user,
                content_type='text',
                content=text[:1000],  # Store first 1000 characters
                result=not is_fake,  # True for real news, False for fake news
                confidence=final_confidence
            )

            return JsonResponse({
                'status': 'success',
                'data': final_analysis,
                'message': 'Text analyzed successfully'
            })

        except Exception as e:
            print(f"Error analyzing text: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'error': f'Error analyzing text: {str(e)}'
            }, status=500)

    return JsonResponse({
        'status': 'error',
        'error': 'Invalid request method'
    }, status=405)

def extract_text_from_url(url):
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Make the request
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text from the body
        text = soup.get_text()
        
        # Clean up the text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Extract title
        title = soup.title.string if soup.title else "Untitled"
        
        # Extract domain for source reliability check
        domain = urlparse(url).netloc
        
        return {
            'text': text,
            'title': title,
            'domain': domain
        }
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing URL content: {str(e)}")
        return None

@login_required(login_url='/login/')
@csrf_protect
def analyze_url(request):
    if request.method == 'POST':
        try:
            url = request.POST.get('url', '').strip()
            if not url:
                return JsonResponse({
                    'status': 'error',
                    'error': 'No URL provided'
                }, status=400)

            # Check if NLTK data is loaded
            if not nltk_data_loaded:
                return JsonResponse({
                    'status': 'error',
                    'error': 'NLTK data not loaded. Please check server logs for details.'
                }, status=500)

            # Check if models are loaded
            if not models_loaded:
                return JsonResponse({
                    'status': 'error',
                    'error': 'Models not loaded. Please check server logs for details.'
                }, status=500)

            # Extract content from URL
            url_content = extract_text_from_url(url)
            if not url_content:
                return JsonResponse({
                    'status': 'error',
                    'error': 'Could not fetch or process the URL content'
                }, status=400)

            # Preprocess and predict with existing models
            processed_text = preprocess_text([url_content['text']])
            vectorized = bert_model.encode(processed_text)

            # Convert to numpy array and ensure it's 2D
            vectorized = np.array(vectorized)
            if len(vectorized.shape) == 1:
                vectorized = vectorized.reshape(1, -1)

            # Get predictions from both models
            xgb_pred = xgb_model.predict(vectorized)[0]
            rf_pred = rf_model.predict(vectorized)[0]

            # Get probabilities from both models
            xgb_proba = xgb_model.predict_proba(vectorized)[0][1]
            rf_proba = rf_model.predict_proba(vectorized)[0][1]

            # Average the predictions
            final_prediction = (xgb_pred + rf_pred) / 2
            final_confidence = (xgb_proba + rf_proba) / 2

            # Determine if it's fake or real
            is_fake = final_prediction < 0.5
            confidence_score = int(final_confidence * 100)

            # Generate initial analysis
            text_analysis = {
                'accuracy': confidence_score,
                'credibility': int((1 - final_confidence) * 100),
                'sentiment': 'Negative' if is_fake else 'Positive',
                'source_reliability': int((1 - final_confidence) * 100),
                'is_fake': int(is_fake)
            }

            # Get Llama model analysis
            llama_analysis = analyze_with_llama(url_content['text'])

            # Combine both analyses
            final_analysis = combine_analysis_results(text_analysis, llama_analysis)

            # Add URL-specific information
            final_analysis['url_info'] = {
                'title': url_content['title'],
                'domain': url_content['domain']
            }

            # Create view history entry
            ViewHistory.objects.create(
                user=request.user,
                content_type='url',
                content=url,
                result=not is_fake,  # True for real news, False for fake news
                confidence=final_confidence
            )

            return JsonResponse({
                'status': 'success',
                'data': final_analysis,
                'message': 'URL analyzed successfully'
            })

        except Exception as e:
            print(f"Error analyzing URL: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'error': f'Error analyzing URL: {str(e)}'
            }, status=500)

    return JsonResponse({
        'status': 'error',
        'error': 'Invalid request method'
    }, status=405)
