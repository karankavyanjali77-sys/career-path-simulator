import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🚀 AI Career Path Simulator")
st.write("Enter your skills and experience to see recommended career paths.")

# Load roles dataset
roles = pd.read_csv("roles.csv")

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Create embeddings once
@st.cache_data
def embed_roles(df):
    df["embedding"] = df["Required_Skills"].apply(
        lambda x: model.encode(x.lower())
    )
    return df

roles = embed_roles(roles)

# User input
skills = st.text_input("Enter your skills (comma separated)")
experience = st.slider("Years of experience",0,10,1)

if st.button("Predict Career Path"):

    if skills.strip()=="":
        st.warning("Please enter skills")
    else:

        user_vector = model.encode(skills.lower())
        role_vectors = np.stack(roles["embedding"].values)

        sims = cosine_similarity([user_vector], role_vectors)[0]
        top = sims.argsort()[-3:][::-1]

        results=[]

        user_list=[s.strip() for s in skills.lower().split(",")]

        for idx in top:
            role=roles.iloc[idx]
            req=[s.strip() for s in role["Required_Skills"].split(",")]

            missing=[s for s in req if s not in user_list]

            roadmap="You are ready!" if not missing else "Learn: "+", ".join(missing[:3])

            predicted=int(role["Avg_Salary"]*(1+experience*0.05))

            results.append({
                "Role":role["Role_Title"],
                "Level":role["Level"],
                "Avg Salary":role["Avg_Salary"],
                "Predicted Salary":predicted,
                "Missing Skills":", ".join(missing) if missing else "None",
                "Roadmap":roadmap
            })

        st.subheader("Top Career Matches")
        st.table(pd.DataFrame(results))
