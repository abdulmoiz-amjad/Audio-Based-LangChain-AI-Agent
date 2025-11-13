# 📝 Audio-Based-LangChain-AI-Agent -- Creative Assistant

## 🚀 Project Overview

This project fine-tunes a **large language model** using **PEFT (Parameter-Efficient Fine-Tuning)** with **LoRA (Low-Rank Adaptation)** and integrates it into a **Streamlit app**.
The app acts as a **creative assistant** to help users generate **stories, recipes, and poetry**, optimized to adapt to **user moods**.

✨ **Key highlights**:

* Fine-tuned model with PEFT + LoRA
* Emotion detection
* Text-to-speech conversion
* Streamlit UI for seamless interaction

---

## 🔑 Key Features

* **Fine-Tuning with PEFT + LoRA**: Efficiently fine-tunes *NousResearch/Llama-2-7b-chat-hf*.
* **Creative Content Generation**: Stories, recipes, and poetry generated from user prompts.
* **Emotion Detection**: Tailored content based on detected mood.
* **Text-to-Speech**: Converts text output into audio with GTTS.
* **Streamlit Integration**: Clean, interactive, and user-friendly interface.

---

## 📂 Project Structure

```bash
├── ai-companion.py        # Streamlit app
├── ai-companion.ipynb     # Notebook version of Streamlit app
├── Fine-tuning.py         # Fine-tuning script
├── Fine-tuning.ipynb      # Notebook version of fine-tuning script
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

> 🔄 Files with the same name but `.py` and `.ipynb` extensions are equivalent (script vs. notebook versions).

---

## 📝 Key Scripts

* **Fine-tuning.py / Fine-tuning.ipynb** – fine-tunes the model using PEFT + LoRA.
* **ai-companion.py / ai-companion.ipynb** – loads the pre-trained + fine-tuned model, integrates with Streamlit, generates content, detects emotions, and handles TTS.

---

## ⚙️ Installation & Setup

### 1️⃣ Install Dependencies

You can use pip directly:

```bash
pip install datasets transformers peft tensorflow langchain langchain-community streamlit gtts pyngrok
```

or install from the requirements file:

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit App

```bash
streamlit run ai-companion.py
```

---

## 🖥 Streamlit App Features

* **User Interaction** – Input a prompt to generate a story, recipe, or poem.
* **Emotion Detection** – Uses DistilBERT classifier to adapt responses.
* **Response Generation** – Tailored text output based on input + detected emotion.
* **Text-to-Speech** – Play generated content as audio.

---

## 🎯 Usage

1. Enter your prompt (story/recipe/poem).
2. Click **Generate Response**.
3. Listen to the generated audio.

**Example Prompts:**

* Story: *"Tell me a story about a brave knight."*
* Recipe: *"Give me a recipe for chocolate cake."*
* Poem: *"Write a poem about love."*

---

## 🧠 Emotion Classifier

* Built using **Hugging Face DistilBERT** model.
* Detects moods like: Joy, Desire, Admiration, Approval, Curiosity, Fear, Sadness, Anger, and Neutral.

---

## 🔊 Text-to-Speech

* Uses **GTTS (Google Text-to-Speech)**.
* Audio playback inside the Streamlit app.

---

## 👥 Authors & Contributors

**Author:**

* Abdulmoiz

**Contributors:**

* Ahsan Waseem
* Wasif Mehboob
* Ameer Hamza

---

## 🙏 Acknowledgments

* [Hugging Face](https://huggingface.co/) for model + libraries.
* [Streamlit](https://streamlit.io/) community for simplifying interactive apps.

---
