# capstone
MSBA Capstone Project

# Capstone Project: Influencer Identification in Healthcare Policy Narratives

This project aims to develop a data-driven framework to identify and rank influential voices shaping healthcare policy narratives across both traditional and social media platforms. Through a combination of data ingestion, sentiment analysis, and influence modeling, the project will surface the origin, amplification, and impact of key healthcare narratives in the U.S.

---

## 👥 Team & Roles

- **Project Manager**: Posy Olivetti  
- **Team Liaison**: Mark Saba  
- **Technical Leads**: Anna Glass, Jasmin Mendoza  
- **Reporting Coordinator**: Mohammad Waqas  

---

## 🔍 Objectives

1. **Influencer Identification Framework**  
   Develop a method to determine who originates vs. amplifies healthcare narratives.

2. **Influence Scoring Model**  
   Quantify influence based on reach, engagement, sentiment, and timing.

3. **Narrative Propagation Analysis**  
   Visualize and track how narratives evolve over time across platforms.

4. **Interactive Tool (Optional)**  
   Deliver a prototype application (e.g., Streamlit) to support real-time insights for stakeholders.

---

## 🧠 Problem Statement

Despite the abundance of healthcare-related content across media, there's no standardized, scalable way to identify who truly drives narrative adoption and how those narratives reach and influence policymakers. This project seeks to fill that gap with a quantitative, repeatable methodology.

---

## 📊 Data Sources (Planned)

- Twitter/X (2020–present, top 30 public health influencers)
- Reddit Threads (e.g., r/Health, r/AskDocs)
- Google Search Trends
- Podcasts (Health-related content)
- YouTube and Streamers
- TV News / Beltway Publications
- TikTok / Instagram Public Posts
- Lobbying Disclosures
- Patent & Trademark Filings
- Media Cloud (open-source news coverage)

---

## 🗂 Project Structure

```plaintext
capstone-influencer-healthcare/
├── data/              ← raw and processed data files
├── notebooks/         ← exploratory notebooks (EDA, modeling)
├── src/               ← all core scripts (ingestion, modeling, etc.)
├── reports/           ← figures and presentation files
├── app/               ← Streamlit or app files
├── README.md
├── requirements.txt
└── .gitignore