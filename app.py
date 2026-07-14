import os
import streamlit as st
from src.emotion_predictor import EmotionPredictor
from src.bert_predictor import BERTEmotionClassifier
from src.mixed_emotion import get_mixed_emotions
from dotenv import load_dotenv
from google import genai
from datetime import datetime
import pandas as pd
import plotly.express as px

load_dotenv()
api_key=os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY not found."); st.stop()
client = genai.Client(api_key=api_key)

@st.cache_resource
def load_models():
    try:
        bilstm_model = EmotionPredictor(
            "models/BiLSTM/emotion_model.keras"
        )

        bert_model = None

        try:
            bert_model = BERTEmotionClassifier()
            bert_model.load_model(
                "models/BERT"
            )
        except Exception as e:
            st.warning(f"BERT model couldn't be loaded: {e}")

        return bilstm_model, bert_model

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None


# Load the models only once
emotion_predictor, bert_classifier = load_models()
#1
if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

def generate_learning_response(field,emotion, confidence, student_text):

    prompt = f"""
You are a helpful learning assistant.

A student studying {field} is feeling {emotion}
(confidence: {confidence:.1%}).

Problem:

{student_text}

Please respond with:

1. Briefly acknowledge the student's feelings.
2. Give one practical study tip specifically for {field}.
3. Suggest the next step they should take.
4. End with encouragement.

Use simple language.
Do not mention confidence scores.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        return response.text.strip()

    except Exception as e:
        return None
    

EMOTION_RESPONSES = {

    "confused": {
        "response":
        "You seem confused. Try breaking the topic into smaller concepts and revisit each one slowly.",
        "action":
        "Review the basics first."
    },

    "frustrated": {
        "response":
        "Learning can be frustrating. Take a short break and approach the problem from another angle.",
        "action":
        "Try another explanation or example."
    },

    "confident": {
        "response":
        "Great work! You're making solid progress.",
        "action":
        "Challenge yourself with a harder problem."
    },

    "bored": {
        "response":
        "The material may not feel engaging right now.",
        "action":
        "Try solving real-world examples."
    },

    "curious": {
        "response":
        "Curiosity is a great sign of learning.",
        "action":
        "Explore advanced resources on this topic."
    }

}

#2 until line 180
def add_to_history(field, problem, emotion, confidence, ai_response,
                   bilstm_scores, bert_result=None):

    def detect_mixed_emotions(scores, threshold=0.15):
        sorted_emotions = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        primary = sorted_emotions[0]
        mixed = [primary]

        for emotion_name, score in sorted_emotions[1:]:
            if score >= threshold:
                mixed.append((emotion_name, score))

        return mixed if len(mixed) > 1 else [primary]

    mixed = detect_mixed_emotions(bilstm_scores)

    emotion_label = (
        " + ".join([e[0] for e in mixed])
        if len(mixed) > 1
        else emotion
    )

    st.session_state.emotion_history.append({
        "timestamp": datetime.now(),
        "field": field,
        "problem": problem,
        "emotion": emotion_label,
        "confidence": confidence,
        "ai_response": ai_response,
        "all_scores": bilstm_scores,
        "model": "BiLSTM"
    })

    if bert_result:

        bert_mixed = detect_mixed_emotions(bert_result["scores"])

        bert_emotion = (
            " + ".join([e[0] for e in bert_mixed])
            if len(bert_mixed) > 1
            else bert_result["emotion"]
        )

        st.session_state.emotion_history.append({
            "timestamp": datetime.now(),
            "field": field,
            "problem": problem,
            "emotion": bert_emotion,
            "confidence": bert_result["confidence"],
            "ai_response": ai_response,
            "all_scores": bert_result["scores"],
            "model": "BERT"
        })

def save_to_csv(field, problem, emotion, confidence, ai_response, model="BiLSTM", all_scores=None):

    try:

        new_example = {
            "text": problem,
            "emotion": emotion.lower(),
            "confidence": confidence,
            "response": ai_response,
            "field": field,
            "model": model,
            "scores": str(all_scores) if all_scores else "",
            "top_score": max(all_scores.values()) if all_scores else confidence,
            "timestamp": datetime.now().isoformat()
        }

        if os.path.exists("emotion_response_examples.csv"):

            df = pd.read_csv("emotion_response_examples.csv")

            df = pd.concat(
                [df, pd.DataFrame([new_example])],
                ignore_index=True
            )

        else:

            df = pd.DataFrame([new_example])

        df.to_csv(
            "emotion_response_examples.csv",
            index=False
        )

        mapping_file = "emotion_response_mapping.csv"

        if os.path.exists(mapping_file):

            mapping_df = pd.read_csv(mapping_file)

        else:

            mapping_df = pd.DataFrame(
                columns=["emotion", "response"]
            )

        duplicate = (
            (mapping_df["emotion"] == emotion) &
            (mapping_df["response"] == ai_response)
        ).any()

        if not duplicate:

            new_mapping = pd.DataFrame([{
                "emotion": emotion,
                "response": ai_response
            }])

            mapping_df = pd.concat(
                [mapping_df, new_mapping],
                ignore_index=True
            )

            mapping_df.to_csv(
                mapping_file,
                index=False
            )

        return True

    except Exception as e:

        st.error(f"Failed to save CSV : {e}")
        return False

@st.cache_data(ttl=60)
def load_interaction_data():
    csv_file = "emotion_response_examples.csv"

    if not os.path.exists(csv_file):
        return pd.DataFrame()

    df = pd.read_csv(csv_file)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df

st.set_page_config(page_title="Emotion Detection & Learning Support Engine",page_icon="🎓",layout="wide")
st.title("🎓 Emotion Detection & Learning Support Engine")
field = st.selectbox(
    "What field are you studying?",
    [
        "Computer Science",
        "Mathematics",
        "Physics",
        "Chemistry",
        "Biology",
        "Engineering",
        "Business",
        "Literature",
        "History",
        "Psychology",
        "Other"
    ]
)

left_col, right_col = st.columns([3, 1])

with left_col:

    problem = st.text_area(
        f"Describe your {field} problem or challenge:",
        placeholder=f"Example: I'm struggling with algorithms in {field}",
        height=220
    )

with right_col:

    with st.container(border=True):

        st.subheader("⚙ Settings")

        use_ai = st.checkbox(
            "Use AI Response (Gemini)",
            value=True
        )

        save_data = st.checkbox(
            "Save to CSV for learning",
            value=True
        )

        show_details = st.checkbox(
            "Show analysis details",
            value=True
        )

st.markdown("---")
st.write("Predict from Saved Data")

use_csv_prediction = st.checkbox(
    "Use CSV-based Prediction",
    value=False
)

examples_df = pd.DataFrame()

if use_csv_prediction:

    if os.path.exists("emotion_response_examples.csv"):

        examples_df = pd.read_csv(
            "emotion_response_examples.csv"
        )

        st.info(
            f"Using {len(examples_df)} saved examples."
        )

    else:

        st.warning(
            "No CSV learning data found."
        )

if st.button("🎓 Get AI Learning Help", use_container_width=True):

    try:

        if not problem.strip():
            st.warning("Please enter some text.")
            st.stop()

        with st.spinner("Analyzing your learning problem..."):

            bilstm_result = emotion_predictor.predict(problem)

            if bert_classifier is not None:
                bert_result = bert_classifier.predict(problem)
            else:
                bert_result = bilstm_result

        emotion_result = bert_result
        emotion = emotion_result["emotion"]
        confidence = emotion_result["confidence"]

        bilstm_mixed = get_mixed_emotions(bilstm_result["scores"])
        bert_mixed = get_mixed_emotions(bert_result["scores"])

        if use_ai:

            gemini_response = generate_learning_response(
                field,
                emotion,
                confidence,
                problem
            )

        else:

            gemini_response = (
                EMOTION_RESPONSES[emotion.lower()]["response"]
                + "\n\nSuggestion: "
                + EMOTION_RESPONSES[emotion.lower()]["action"]
            )

        # Fallback to templates if Gemini failed
        if use_ai and not gemini_response:

            gemini_response = (
                EMOTION_RESPONSES[bert_result["emotion"].lower()]["response"]
                + "\n\nSuggestion: "
                + EMOTION_RESPONSES[bert_result["emotion"].lower()]["action"]
            )

        add_to_history(
            field=field,
            problem=problem,
            emotion=emotion,
            confidence=confidence,
            ai_response=gemini_response,
            bilstm_scores=bilstm_result["scores"],
            bert_result=bert_result
        )

        if save_data:

            save_to_csv(
                field=field,
                problem=problem,
                emotion=bilstm_result["emotion"],
                confidence=bilstm_result["confidence"],
                ai_response=gemini_response,
                model="BiLSTM",
                all_scores=bilstm_result["scores"]
            )

            if bert_classifier is not None:

                save_to_csv(
                    field=field,
                    problem=problem,
                    emotion=bert_result["emotion"],
                    confidence=bert_result["confidence"],
                    ai_response=gemini_response,
                    model="BERT",
                    all_scores=bert_result["scores"]
                )

            st.success("Interaction(s) saved for future learning.")

        st.subheader("Model Predictions Comparison")

        if bert_classifier is not None:
            col1, col2 = st.columns(2)
        else:
            col1 = st.columns(1)[0]
            col2 = None

        with col1:

            st.write("BiLSTM Student Adaptive")

            bilstm_text = " + ".join(
                f"{e.capitalize()}"
                for e, _ in bilstm_mixed
            )

            st.metric(
                "Mixed Emotions" if len(bilstm_mixed) > 1 else "Emotion",
                bilstm_text if len(bilstm_mixed) > 1 else bilstm_result["emotion"].capitalize(),
                f"{bilstm_result['confidence']:.1%}"
            )

            for e, s in sorted(
                bilstm_result["scores"].items(),
                key=lambda x: x[1],
                reverse=True
            ):
                st.progress(
                    float(s),
                    text=f"{e.capitalize()}: {s:.1%}"
                )

        if col2 is not None:

            with col2:

                st.write("BERT Transformer")

                bert_text = " + ".join(
                    f"{e.capitalize()}"
                    for e, _ in bert_mixed
                )

                st.metric(
                    "Mixed Emotions" if len(bert_mixed) > 1 else "Emotion",
                    bert_text if len(bert_mixed) > 1 else bert_result["emotion"].capitalize(),
                    f"{bert_result['confidence']:.1%}"
                )

                for e, s in sorted(
                    bert_result["scores"].items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    st.progress(
                        float(s),
                        text=f"{e.capitalize()}: {s:.1%}"
                    )

        st.markdown("---")
        st.header("Learning Analytics")

        history_df = load_interaction_data()

        if not history_df.empty:
            history_df = pd.DataFrame(st.session_state.emotion_history)
            tab1, tab2, tab3 = st.tabs(
                ["Emotions", "Fields", "Summary"]
            )

            with tab1:
                col1, col2 = st.columns(2)

                with col1:
                    emotion_counts = history_df["emotion"].value_counts()
                    fig1 = px.pie(
                        values=emotion_counts.values,
                        names=emotion_counts.index,
                        title="Emotion Distribution"
                    )

                    st.plotly_chart(fig1, use_container_width=True)

                with col2:
                    df_copy = history_df.copy()

                    if "timestamp" in df_copy.columns:
                        df_copy = df_copy.sort_values("timestamp")
                        df_copy["time"] = df_copy["timestamp"].dt.strftime("%H:%M:%S")
                    else:
                        df_copy["time"] = range(len(df_copy))

                    fig2 = px.line(
                        df_copy,
                        x="time",
                        y="confidence",
                        color="emotion",
                        markers=True,
                        title="Emotional Journey"
                    )

                    st.plotly_chart(fig2, use_container_width=True)
            with tab2:
                if "model" in history_df.columns:
                    field_summary = (
                        history_df
                        .groupby(["field", "emotion", "model"])
                        .size()
                        .reset_index(name="count")
                    )

                    fig3 = px.bar(
                        field_summary,
                        x="field",
                        y="count",
                        color="emotion",
                        facet_col="model",
                        title="Emotions by Study Field & Model"
                    )
                else:
                    field_summary = (
                        history_df
                        .groupby(["field", "emotion"])
                        .size()
                        .reset_index(name="count")
                    )
                    
                    fig3 = px.bar(
                        field_summary,
                        x="field",
                        y="count",
                        color="emotion",
                        title="Emotions by Study Field"
                  )

                st.plotly_chart(fig3, use_container_width=True)

            with tab3:

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Total Sessions",
                        len(history_df)
                    )

                with c2:
                    st.metric(
                        "Average Confidence",
                        f"{history_df['confidence'].mean():.1%}"
                    )

                with c3:
                    st.metric(
                        "Models Used",
                        history_df["model"].nunique()
                    )

                st.dataframe(
                    history_df.tail(10),
                    use_container_width=True
                )

        st.divider()

        st.subheader("🎓 Personalized Learning Support")

        st.success(gemini_response)

    except Exception as e:

        st.error(f"Error: {e}")
# Sidebar Dashboard
with st.sidebar:
    st.header("📊 Dashboard")
    status = "BiLSTM"
    if bert_classifier:
        status += " + BERT"
    st.write(f"Models Loaded: {status}")
    st.write(f"BiLSTM Loaded: {'✅'}")
    st.write(f"BERT Loaded: {'✅' if bert_classifier else '❌'}")
    st.write(
        f"Total Interactions: {len(st.session_state.emotion_history)}"
    )

    if os.path.exists("emotion_response_examples.csv"):
        examples_df = pd.read_csv(
            "emotion_response_examples.csv"
        )
        st.write(
            f"CSV Examples: {len(examples_df)}"
        )
    else:
        st.write("CSV Examples: 0")
        
    if st.button("Clear History"):
        st.session_state.emotion_history = []

        st.rerun()

    if st.session_state.emotion_history:
        st.subheader("Recent Sessions")
        recent = st.session_state.emotion_history[-3:]

        for item in reversed(recent):
            ts = item["timestamp"].strftime("%H:%M") if hasattr(item["timestamp"], "strftime") else str(item["timestamp"])
            st.write(
                f"🕒 {ts} | 📘 {item['field']} | 🤖 {item['model']} | {item['emotion']} ({item['confidence']:.1%})"
            )
