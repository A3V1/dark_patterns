from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import requests
from bs4 import BeautifulSoup, Tag
import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import urlparse, urljoin
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load tokenizer & model once at startup
try:
    tokenizer = BertTokenizer.from_pretrained("./saved_model")
    model = BertForSequenceClassification.from_pretrained("./saved_model")
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    tokenizer = None
    model = None

app = FastAPI(
    title="Enhanced Dark Pattern Detector API",
    description="Comprehensive API for detecting dark patterns in web content using ML and enhanced UI detection",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schemas
class TextAnalysisRequest(BaseModel):
    texts: List[str]

class URLAnalysisRequest(BaseModel):
    url: str

class CombinedAnalysisRequest(BaseModel):
    url: Optional[str] = None
    texts: Optional[List[str]] = None

# Label mapping
LABEL_MAP = {
    0: "Social Proof",
    1: "Scarcity",
    2: "Obstruction",
    3: "Sneaking",
    4: "Forced Action",
    5: "Misdirection",
    6: "Nagging",
    7: "Urgency",
    8: "Not Dark Pattern"
}

@dataclass
class DetectedPattern:
    name: str
    category: str
    type: str
    confidence: float
    description: str
    element: Optional[str] = None
    severity: str = "medium"  # low, medium, high, critical

class EnhancedUIPatternDetector:
    """Enhanced detector for UI-based dark patterns with better accuracy and coverage"""
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self) -> List[Dict]:
        """Initialize comprehensive dark pattern detection rules"""
        return [
            # OBSTRUCTION PATTERNS
            {
                "name": "Pre-checked consent checkboxes",
                "category": "Obstruction",
                "severity": "high",
                "description": "Checkboxes for subscriptions/consents that are pre-checked",
                "detectors": [
                    self._detect_prechecked_consents,
                ]
            },
            {
                "name": "Hidden unsubscribe links",
                "category": "Obstruction", 
                "severity": "high",
                "description": "Unsubscribe links that are hard to find or see",
                "detectors": [
                    self._detect_hidden_unsubscribe,
                ]
            },
            {
                "name": "Difficult account deletion",
                "category": "Obstruction",
                "severity": "critical",
                "description": "Account deletion options that are buried or hard to access",
                "detectors": [
                    self._detect_difficult_deletion,
                ]
            },
            {
                "name": "Privacy policy barriers",
                "category": "Obstruction",
                "severity": "medium",
                "description": "Privacy settings that are hard to access or understand",
                "detectors": [
                    self._detect_privacy_barriers,
                ]
            },
            
            # SCARCITY PATTERNS
            {
                "name": "Fake countdown timers",
                "category": "Scarcity",
                "severity": "high", 
                "description": "Countdown timers that create false urgency",
                "detectors": [
                    self._detect_countdown_timers,
                ]
            },
            {
                "name": "Low stock warnings",
                "category": "Scarcity",
                "severity": "medium",
                "description": "Claims about limited quantities or stock",
                "detectors": [
                    self._detect_stock_warnings,
                ]
            },
            {
                "name": "High demand claims",
                "category": "Scarcity",
                "severity": "medium",
                "description": "Claims about popularity or high demand",
                "detectors": [
                    self._detect_demand_claims,
                ]
            },
            {
                "name": "Limited time offers",
                "category": "Scarcity",
                "severity": "medium",
                "description": "Time-sensitive offers that may be misleading",
                "detectors": [
                    self._detect_limited_offers,
                ]
            },
            
            # SOCIAL PROOF PATTERNS
            {
                "name": "Fake activity notifications",
                "category": "Social Proof",
                "severity": "high",
                "description": "Notifications about other users' actions that may be fabricated",
                "detectors": [
                    self._detect_fake_notifications,
                ]
            },
            {
                "name": "Manipulated reviews",
                "category": "Social Proof", 
                "severity": "high",
                "description": "Review systems that appear manipulated",
                "detectors": [
                    self._detect_manipulated_reviews,
                ]
            },
            {
                "name": "Fake testimonials",
                "category": "Social Proof",
                "severity": "medium",
                "description": "Testimonials that appear fabricated",
                "detectors": [
                    self._detect_fake_testimonials,
                ]
            },
            
            # FORCED ACTION PATTERNS
            {
                "name": "Forced registration",
                "category": "Forced Action",
                "severity": "high",
                "description": "Modals or popups that force sign-up to continue",
                "detectors": [
                    self._detect_forced_registration,
                ]
            },
            {
                "name": "Subscription traps",
                "category": "Forced Action",
                "severity": "critical",
                "description": "Free trials that automatically convert to paid subscriptions",
                "detectors": [
                    self._detect_subscription_traps,
                ]
            },
            {
                "name": "Forced sharing",
                "category": "Forced Action",
                "severity": "medium", 
                "description": "Requirements to share content to access features",
                "detectors": [
                    self._detect_forced_sharing,
                ]
            },
            
            # SNEAKING PATTERNS
            {
                "name": "Hidden costs",
                "category": "Sneaking",
                "severity": "critical",
                "description": "Additional costs that are hidden or disclosed late",
                "detectors": [
                    self._detect_hidden_costs,
                ]
            },
            {
                "name": "Bait and switch pricing",
                "category": "Sneaking",
                "severity": "high",
                "description": "Prices that change or have hidden conditions",
                "detectors": [
                    self._detect_bait_switch,
                ]
            },
            {
                "name": "Sneak into basket",
                "category": "Sneaking",
                "severity": "high",
                "description": "Additional items added to cart without clear consent",
                "detectors": [
                    self._detect_sneak_basket,
                ]
            },
            
            # MISDIRECTION PATTERNS
            {
                "name": "Confusing button hierarchy",
                "category": "Misdirection", 
                "severity": "medium",
                "description": "Buttons designed to mislead user intent",
                "detectors": [
                    self._detect_button_misdirection,
                ]
            },
            {
                "name": "Misleading labels",
                "category": "Misdirection",
                "severity": "medium",
                "description": "Form labels or buttons with misleading text",
                "detectors": [
                    self._detect_misleading_labels,
                ]
            },
            {
                "name": "Disguised ads",
                "category": "Misdirection",
                "severity": "medium",
                "description": "Advertisements disguised as content",
                "detectors": [
                    self._detect_disguised_ads,
                ]
            },
            
            # NAGGING PATTERNS
            {
                "name": "Persistent popups",
                "category": "Nagging",
                "severity": "medium",
                "description": "Popups that repeatedly appear despite dismissal",
                "detectors": [
                    self._detect_persistent_popups,
                ]
            },
            {
                "name": "Auto-playing media",
                "category": "Nagging",
                "severity": "medium",
                "description": "Videos or audio that play automatically",
                "detectors": [
                    self._detect_autoplay_media,
                ]
            },
            {
                "name": "Excessive notifications",
                "category": "Nagging",
                "severity": "low",
                "description": "Requests for notification permissions or excessive alerts",
                "detectors": [
                    self._detect_notification_spam,
                ]
            },
        ]
    
    def detect_patterns(self, soup: BeautifulSoup, html: str, url: str = "") -> List[DetectedPattern]:
        """Main method to detect all UI patterns"""
        detected_patterns = []
        
        for pattern_config in self.patterns:
            try:
                for detector in pattern_config["detectors"]:
                    result = detector(soup, html, url)
                    if result:
                        confidence, element, description = result if isinstance(result, tuple) else (0.8, None, pattern_config["description"])
                        
                        detected_patterns.append(DetectedPattern(
                            name=pattern_config["name"],
                            category=pattern_config["category"],
                            type="UI Pattern",
                            confidence=confidence,
                            description=description or pattern_config["description"],
                            element=element,
                            severity=pattern_config["severity"]
                        ))
                        break  # Only report each pattern once
                        
            except Exception as e:
                logger.warning(f"Error detecting pattern {pattern_config['name']}: {e}")
        
        return detected_patterns
    
    # OBSTRUCTION DETECTORS
    def _detect_prechecked_consents(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect pre-checked checkboxes for consents/subscriptions"""
        consent_keywords = r'\b(newsletter|marketing|promotional|consent|agree|terms|privacy|subscribe|updates)\b'
        
        checkboxes = soup.find_all("input", {"type": "checkbox", "checked": True})
        for checkbox in checkboxes:
            # Check nearby text for consent-related keywords
            parent = checkbox.find_parent()
            if parent:
                text = parent.get_text(strip=True).lower()
                if re.search(consent_keywords, text, re.I):
                    return (0.9, str(checkbox)[:100], "Pre-checked consent checkbox found")
        
        return None
    
    def _detect_hidden_unsubscribe(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect hidden or hard-to-find unsubscribe links"""
        unsubscribe_links = soup.find_all("a", string=re.compile(r"unsubscribe", re.I))
        
        for link in unsubscribe_links:
            style = link.get("style", "")
            classes = " ".join(link.get("class", []))
            
            # Check if link is visually hidden
            if any(hidden in style.lower() for hidden in ["display:none", "visibility:hidden"]) or \
               any(hidden in classes.lower() for hidden in ["hidden", "invisible"]):
                return (0.95, str(link)[:100], "Hidden unsubscribe link detected")
            
            # Check if text is very small
            if re.search(r'font-size:\s*[0-9]px', style) or "tiny" in classes.lower():
                return (0.8, str(link)[:100], "Unsubscribe link with very small text")
        
        return None
    
    def _detect_difficult_deletion(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect difficult account deletion patterns"""
        deletion_keywords = r'\b(delete.*account|close.*account|deactivate|remove.*account)\b'
        
        # Look for deletion links that are buried or require multiple steps
        text_lower = html.lower()
        if re.search(deletion_keywords, text_lower, re.I):
            # Check if deletion is mentioned but requires contacting support
            if re.search(r'contact.*support.*delete|email.*us.*delete', text_lower, re.I):
                return (0.8, None, "Account deletion requires contacting support")
            
            # Check for deletion buried in settings/privacy pages
            if re.search(r'settings.*delete|privacy.*delete', text_lower, re.I):
                return (0.6, None, "Account deletion buried in settings")
        
        return None
    
    def _detect_privacy_barriers(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect privacy settings that are hard to access"""
        privacy_elements = soup.find_all(string=re.compile(r"privacy.*settings|manage.*privacy", re.I))
        
        for element in privacy_elements:
            parent = element.find_parent() if hasattr(element, 'find_parent') else element.parent
            if parent and parent.name == 'a':
                # Check if privacy link requires multiple clicks or is buried
                href = parent.get('href', '')
                if 'settings' in href and 'privacy' in href:
                    return (0.6, str(parent)[:100], "Privacy settings require navigation through multiple pages")
        
        return None
    
    # SCARCITY DETECTORS
    def _detect_countdown_timers(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect countdown timers and urgency indicators"""
        # Look for timer elements
        timer_selectors = ["#countdown", "[id*='timer']", "[class*='countdown']", "[class*='timer']"]
        for selector in timer_selectors:
            elements = soup.select(selector)
            if elements:
                return (0.9, str(elements[0])[:100], "Countdown timer detected")
        
        # Look for urgency text patterns
        urgency_patterns = [
            r'\b(expires?\s+in|ends?\s+in|\d+\s*:\s*\d+\s*:\s*\d+)\b',
            r'\b(hurry|act\s+fast|limited\s+time|time\s+running\s+out)\b',
            r'\b(sale\s+ends|offer\s+expires|deadline)\b'
        ]
        
        for pattern in urgency_patterns:
            if re.search(pattern, html, re.I):
                return (0.7, None, "Urgency language detected")
        
        return None
    
    def _detect_stock_warnings(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect low stock or inventory warnings"""
        stock_patterns = [
            r'\b(only\s+\d+\s+left|last\s+\d+|limited\s+stock)\b',
            r'\b(almost\s+gone|nearly\s+sold\s+out|low\s+stock)\b',
            r'\b(\d+\s+in\s+stock|\d+\s+remaining)\b'
        ]
        
        for pattern in stock_patterns:
            matches = re.finditer(pattern, html, re.I)
            for match in matches:
                return (0.8, match.group(), "Low stock warning detected")
        
        return None
    
    def _detect_demand_claims(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect high demand or popularity claims"""
        demand_patterns = [
            r'\b(\d+\s+people\s+viewing|others\s+are\s+looking)\b',
            r'\b(\d+\s+bought.*last|popular\s+choice|trending)\b',
            r'\b(high\s+demand|selling\s+fast|most\s+popular)\b'
        ]
        
        for pattern in demand_patterns:
            matches = re.finditer(pattern, html, re.I)
            for match in matches:
                return (0.7, match.group(), "High demand claim detected")
        
        return None
    
    def _detect_limited_offers(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect limited time offers"""
        offer_patterns = [
            r'\b(limited\s+time\s+offer|special\s+offer|today\s+only)\b',
            r'\b(flash\s+sale|daily\s+deal|exclusive\s+offer)\b'
        ]
        
        for pattern in offer_patterns:
            if re.search(pattern, html, re.I):
                return (0.6, None, "Limited time offer detected")
        
        return None
    
    # SOCIAL PROOF DETECTORS
    def _detect_fake_notifications(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect potentially fake activity notifications"""
        notification_patterns = [
            r'\b(someone\s+just\s+bought|\w+\s+from\s+\w+\s+just\s+purchased)\b',
            r'\b(\d+\s+people\s+bought.*today|new\s+customer.*just\s+signed)\b'
        ]
        
        # Look for notification-style elements
        notification_selectors = [
            "[class*='notification']", "[class*='alert']", "[class*='popup']",
            "[class*='toast']", "[id*='notification']"
        ]
        
        for selector in notification_selectors:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                for pattern in notification_patterns:
                    if re.search(pattern, text, re.I):
                        return (0.8, str(element)[:100], "Suspicious activity notification")
        
        return None
    
    def _detect_manipulated_reviews(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect potentially manipulated review systems"""
        review_elements = soup.find_all(class_=re.compile(r"review|rating|testimonial", re.I))
        
        if len(review_elements) > 3:  # Only check if there are enough reviews
            # Look for suspiciously similar review patterns
            review_texts = [elem.get_text(strip=True) for elem in review_elements]
            
            # Check for very similar lengths (indicator of fake reviews)
            lengths = [len(text) for text in review_texts if len(text) > 10]
            if len(lengths) > 3:
                avg_length = sum(lengths) / len(lengths)
                similar_length_count = sum(1 for length in lengths if abs(length - avg_length) < 10)
                if similar_length_count / len(lengths) > 0.7:
                    return (0.7, None, "Reviews have suspiciously similar lengths")
        
        return None
    
    def _detect_fake_testimonials(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect potentially fake testimonials"""
        testimonial_elements = soup.find_all(class_=re.compile(r"testimonial|review", re.I))
        
        fake_indicators = [
            r'\b(changed\s+my\s+life|best\s+decision|amazing\s+results)\b',
            r'\b(highly\s+recommend|five\s+stars|incredible)\b'
        ]
        
        for element in testimonial_elements:
            text = element.get_text(strip=True)
            for pattern in fake_indicators:
                if re.search(pattern, text, re.I):
                    return (0.6, str(element)[:100], "Potentially exaggerated testimonial")
        
        return None
    
    # FORCED ACTION DETECTORS
    def _detect_forced_registration(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect forced registration modals"""
        modal_selectors = [
            "[class*='modal']", "[class*='popup']", "[class*='overlay']",
            "[id*='modal']", "[id*='popup']"
        ]
        
        for selector in modal_selectors:
            modals = soup.select(selector)
            for modal in modals:
                text = modal.get_text(strip=True).lower()
                if any(keyword in text for keyword in ['sign up', 'register', 'create account']) and \
                   any(keyword in text for keyword in ['continue', 'access', 'view']):
                    return (0.9, str(modal)[:100], "Forced registration modal detected")
        
        return None
    
    def _detect_subscription_traps(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect subscription traps in free trials"""
        trial_keywords = r'\b(free\s+trial|trial\s+period|no\s+commitment)\b'
        billing_keywords = r'\b(billing|charged|subscription|auto-renew)\b'
        
        if re.search(trial_keywords, html, re.I):
            # Look for subscription terms in fine print
            fine_print_selectors = ["small", "[class*='fine']", "[class*='terms']", "[style*='font-size']"]
            
            for selector in fine_print_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    if re.search(billing_keywords, text, re.I):
                        return (0.8, str(element)[:100], "Free trial with automatic billing in fine print")
        
        return None
    
    def _detect_forced_sharing(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect forced social sharing requirements"""
        sharing_patterns = [
            r'\b(share\s+to\s+unlock|tweet\s+to\s+access|post\s+to\s+continue)\b',
            r'\b(like\s+to\s+download|follow\s+to\s+access)\b'
        ]
        
        for pattern in sharing_patterns:
            if re.search(pattern, html, re.I):
                return (0.8, None, "Forced social sharing detected")
        
        return None
    
    # SNEAKING DETECTORS
    def _detect_hidden_costs(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect hidden additional costs"""
        cost_patterns = [
            r'\*.*(?:additional|extra|shipping|handling|processing|tax|fee)',
            r'\b(plus\s+(?:tax|shipping|fees)|excluding\s+(?:tax|shipping))\b'
        ]
        
        for pattern in cost_patterns:
            matches = re.finditer(pattern, html, re.I)
            for match in matches:
                # Check if this cost info is in small text or hidden
                surrounding_text = html[max(0, match.start()-100):match.end()+100]
                if '<small>' in surrounding_text or 'font-size' in surrounding_text:
                    return (0.9, match.group(), "Hidden additional costs in fine print")
        
        return None
    
    def _detect_bait_switch(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect bait and switch pricing tactics"""
        price_elements = soup.find_all(string=re.compile(r'\$\d+|\d+\.\d{2}', re.I))
        
        if len(price_elements) > 2:
            # Look for "starting at" or "from" prices that may be misleading
            bait_patterns = [
                r'\b(starting\s+(?:at|from)|from\s+(?:only|\$)|\*price\s+shown)\b',
                r'\b(as\s+low\s+as|prices\s+from)\b'
            ]
            
            for pattern in bait_patterns:
                if re.search(pattern, html, re.I):
                    return (0.7, None, "Potentially misleading 'starting at' pricing")
        
        return None
    
    def _detect_sneak_basket(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect items sneaked into shopping basket"""
        # Look for pre-checked add-on items
        checkboxes = soup.find_all("input", {"type": "checkbox", "checked": True})
        
        for checkbox in checkboxes:
            parent = checkbox.find_parent()
            if parent:
                text = parent.get_text(strip=True).lower()
                if any(keyword in text for keyword in ['warranty', 'insurance', 'protection', 'upgrade', 'add-on']):
                    return (0.8, str(checkbox)[:100], "Pre-selected additional items")
        
        return None
    
    # MISDIRECTION DETECTORS
    def _detect_button_misdirection(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect confusing button designs that mislead users"""
        buttons = soup.find_all(["button", "input", "a"], class_=re.compile(r"btn|button", re.I))
        
        positive_buttons = []
        negative_buttons = []
        
        for button in buttons:
            text = button.get_text(strip=True).lower()
            
            # Classify buttons as positive or negative actions
            if any(word in text for word in ['yes', 'accept', 'continue', 'agree', 'subscribe']):
                positive_buttons.append(button)
            elif any(word in text for word in ['no', 'cancel', 'decline', 'skip', 'unsubscribe']):
                negative_buttons.append(button)
        
        # Check if negative buttons are styled like positive ones or vice versa
        if positive_buttons and negative_buttons:
            # This is a simplified check - in reality, you'd want to analyze CSS styles
            for neg_btn in negative_buttons:
                classes = " ".join(neg_btn.get("class", []))
                if any(pos_class in classes for pos_class in ['primary', 'success', 'green']):
                    return (0.7, str(neg_btn)[:100], "Negative action button styled as positive")
        
        return None
    
    def _detect_misleading_labels(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect misleading form labels or button text"""
        misleading_patterns = [
            (r'\b(no\s+thanks.*continue|skip.*accept)\b', "Confusing negative option"),
            (r'\b(yes.*spam|accept.*marketing)\b', "Misleading consent language"),
        ]
        
        for pattern, description in misleading_patterns:
            if re.search(pattern, html, re.I):
                return (0.6, None, description)
        
        return None
    
    def _detect_disguised_ads(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect advertisements disguised as content"""
        # Look for elements that might be disguised ads
        suspicious_elements = soup.find_all(class_=re.compile(r"sponsored|promoted|ad", re.I))
        
        for element in suspicious_elements:
            # Check if the ad disclosure is hidden or unclear
            ad_text = element.get_text(strip=True)
            if len(ad_text) > 50 and not re.search(r'\b(ad|sponsored|promoted)\b', ad_text[:20], re.I):
                return (0.6, str(element)[:100], "Potentially disguised advertisement")
        
        return None
    
    # NAGGING DETECTORS
    def _detect_persistent_popups(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect popups that may be persistent or hard to dismiss"""
        popup_selectors = ["[class*='popup']", "[class*='modal']", "[id*='popup']"]
        
        for selector in popup_selectors:
            popups = soup.select(selector)
            for popup in popups:
                # Look for popups without clear close buttons
                close_button = popup.find(class_=re.compile(r"close|dismiss", re.I))
                if not close_button:
                    close_button = popup.find(string=re.compile(r"[×✕]|close", re.I))
                
                if not close_button:
                    return (0.7, str(popup)[:100], "Popup without clear close button")
        
        return None
    
    def _detect_autoplay_media(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect auto-playing media"""
        autoplay_video = soup.find("video", {"autoplay": True})
        autoplay_audio = soup.find("audio", {"autoplay": True})
        
        if autoplay_video:
            return (0.9, str(autoplay_video)[:100], "Auto-playing video detected")
        if autoplay_audio:
            return (0.9, str(autoplay_audio)[:100], "Auto-playing audio detected")
        
        return None
    
    def _detect_notification_spam(self, soup: BeautifulSoup, html: str, url: str) -> Optional[Tuple]:
        """Detect excessive notification requests"""
        notification_patterns = [
            r'\b(enable\s+notifications|allow\s+notifications|turn\s+on\s+notifications)\b',
            r'\b(push\s+notifications|browser\s+notifications)\b'
        ]
        
        notification_count = 0
        for pattern in notification_patterns:
            matches = re.findall(pattern, html, re.I)
            notification_count += len(matches)
        
        if notification_count > 2:  # Multiple notification requests
            return (0.6, None, "Multiple notification permission requests")
        
        return None

# Initialize the enhanced detector
ui_detector = EnhancedUIPatternDetector()

def predict_text_ml(text: str) -> Dict[str, Any]:
    if not model or not tokenizer:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=128
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_label = torch.argmax(probs, dim=1).item()
    confidence = probs.max().item()
    
    return {
        "text": text,
        "label": int(pred_label),
        "category": LABEL_MAP[pred_label],
        "confidence": float(confidence),
        "probabilities": {
            LABEL_MAP[i]: float(prob) for i, prob in enumerate(probs.squeeze().tolist())
        }
    }


def fetch_url_content(url: str):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DarkPatternDetector/1.0)"}
    res = requests.get(url, headers=headers, timeout=15)
    if res.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: HTTP {res.status_code}")
    html = res.text
    soup = BeautifulSoup(html, "html.parser")
    return html, soup

def extract_text_elements(soup: BeautifulSoup):
    for script in soup(["script", "style"]):
        script.decompose()
    selectors = ["p", "a", "button", "span", "h1", "h2", "h3", "label", "div[class*='text']"]
    texts = []
    for selector in selectors:
        for elem in soup.select(selector):
            text = elem.get_text(strip=True)
            if 3 < len(text) < 500:
                texts.append(text)
    return list(dict.fromkeys(texts))

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None and tokenizer is not None}

@app.post("/predict/text")
def predict_text(request: TextAnalysisRequest):
    results = [predict_text_ml(t) for t in request.texts if t.strip()]
    return {"success": True, "total_texts": len(results), "predictions": results}


def analyze_url_content(url: str):
    html, soup = fetch_url_content(url)
    texts = extract_text_elements(soup)
    all_ml_predictions = [predict_text_ml(t) for t in texts[:350]]
    ui_patterns = ui_detector.detect_patterns(soup, html, url)

    dark_pattern_ml_predictions = [pred for pred in all_ml_predictions if pred["category"] != "Not Dark Pattern"]

    category_counts = {}
    high_conf = []
    for pred in dark_pattern_ml_predictions:
        cat = pred["category"]
        conf = pred["confidence"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        if conf > 0.7:
            high_conf.append(pred)

    return {
        "success": True,
        "url": url,
        "analysis": {
            "total_texts_analyzed": len(all_ml_predictions), # Only count dark patterns
            "ui_patterns": ui_patterns,
            "ml_predictions": {
                "high_confidence": high_conf,
                "category_distribution": category_counts,
                "all_predictions": dark_pattern_ml_predictions # Only return dark patterns
            }
        },
        "summary": {
            "ui_patterns_found": len(ui_patterns),
            "high_confidence_ml_detections": len(high_conf),
            "most_common_category": max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None
        }
    }

@app.post("/predict/url")
def predict_url(request: URLAnalysisRequest):
    return analyze_url_content(request.url)
    
@app.get("/predict/url")
def predict_url_get(url: str):
    return analyze_url_content(url)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
