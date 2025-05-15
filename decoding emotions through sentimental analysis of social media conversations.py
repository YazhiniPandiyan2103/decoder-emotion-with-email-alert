# -- coding: utf-8 --

import re
import torch
import nltk
import smtplib
import logging
import matplotlib.pyplot as plt
from collections import Counter, deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import tkinter as tk
from tkinter import messagebox, scrolledtext

# Download NLTK tokenizer
nltk.download('punkt', quiet=True)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load model and tokenizer
model_name = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Emotion map
emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
emotion_to_sentiment = {
    "anger": "negative", "disgust": "negative", "fear": "negative",
    "joy": "positive", "neutral": "neutral", "sadness": "negative",
    "surprise": "positive"
}
emotion_emojis = {
    "joy": u"\U0001F604",        # 😄
    "sadness": u"\U0001F622",    # 😢
    "anger": u"\U0001F620",      # 😠
    "fear": u"\U0001F628",       # 😨
    "disgust": u"\U0001F922",    # 🤢
    "surprise": u"\U0001F632",   # 😲
    "neutral": u"\U0001F610",    # 😐
}

# Email config
SENDER_EMAIL = "yazhinipandi2006@gmail.com"
SENDER_PASSWORD = "qxon yhgn fxmm ynbc"
RECIPIENT_EMAIL = "yazhinipandi2006@gmail.com"
NEGATIVE_THRESHOLD = 3
negative_messages = deque(maxlen=NEGATIVE_THRESHOLD)  # Store last 3 negative messages

emotion_counter = Counter()

# Text cleaning
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Analyze emotion
def analyze_emotion(text):
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    pred_id = torch.argmax(probs).item()
    pred_emotion = emotion_labels[pred_id]
    confidence = probs[0][pred_id].item()
    sentiment = emotion_to_sentiment[pred_emotion]
    emotion_counter[pred_emotion] += 1
    return pred_emotion, round(confidence * 100, 2), sentiment

# Send email
def send_email_alert(subject, body, to_email):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        logging.info("Email alert sent.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

# Plot chart
def show_emotion_chart():
    if not emotion_counter:
        messagebox.showinfo("No Data", "No emotions to display.")
        return
    emotions = list(emotion_counter.keys())
    counts = list(emotion_counter.values())
    plt.figure(figsize=(8, 5))
    plt.bar(emotions, counts, color='lightgreen')
    plt.xlabel("Emotions")
    plt.ylabel("Count")
    plt.title("Emotion Frequency Chart")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# GUI Functions
def on_analyze():
    text = input_text.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Error", "Please enter some text.")
        return

    # Split into sentences (line by line analysis)
    lines = text.splitlines()
    for line in lines:
        if not line.strip():
            continue
        emotion, confidence, sentiment = analyze_emotion(line)
        emoji = emotion_emojis.get(emotion, "")

        output = (
            f"Text: {line}\n"
            f"Emotion: {emotion} ({confidence}%) {emoji}\n"
            f"Sentiment: {sentiment}\n"
            f"{'-'*40}\n"
        )
        result_box.insert(tk.END, output)
        result_box.see(tk.END)

        if sentiment == "negative":
            negative_messages.append(line)

            if len(negative_messages) >= NEGATIVE_THRESHOLD:
                combined = "\n".join(negative_messages)
                send_email_alert(
                    subject="Alert: Repeated Negative Messages",
                    body=f"{NEGATIVE_THRESHOLD} negative messages detected:\n\n{combined}",
                    to_email=RECIPIENT_EMAIL
                )
                messagebox.showinfo("Alert", "Email alert sent for repeated negative messages.")
                negative_messages.clear()
        else:
            negative_messages.clear()

# Build GUI
root = tk.Tk()
root.title("Emotion Detector")

tk.Label(root, text="Enter your message below:").pack(pady=5)
input_text = scrolledtext.ScrolledText(root, width=60, height=4)
input_text.pack()

tk.Button(root, text="Analyze Emotion", command=on_analyze, bg="lightblue").pack(pady=10)

tk.Label(root, text="Results:").pack()
result_box = scrolledtext.ScrolledText(root, width=60, height=10)
result_box.pack()

# Optional: Large emoji label (can remove if not needed)
emoji_label = tk.Label(root, text="", font=("Arial", 40))
emoji_label.pack(pady=5)

tk.Button(root, text="Show Emotion Chart", command=show_emotion_chart, bg="lightgreen").pack(pady=10)
tk.Button(root, text="Exit", command=root.destroy, bg="lightcoral").pack(pady=5)

root.mainloop()
