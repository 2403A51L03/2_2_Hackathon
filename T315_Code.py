import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request, jsonify
import spacy

# Load NLP Model
nlp = spacy.load("en_core_web_sm")

# Flask App
app = Flask(__name__)

# Sample Career Data
career_data = pd.DataFrame({
    'career': [
        'Software Engineer', 'Data Scientist', 'Doctor', 'Teacher', 'Cybersecurity Analyst',
        'AI Engineer', 'Mechanical Engineer', 'Marketing Manager', 'Graphic Designer', 'Civil Engineer',
        'Financial Analyst', 'Psychologist', 'UX/UI Designer', 'Product Manager', 'Biomedical Scientist',
        'Network Engineer', 'Content Writer', 'HR Manager', 'Architect', 'Entrepreneur'
    ],
    'required_skills': [
        'programming, problem-solving, algorithms',
        'statistics, machine learning, data analysis',
        'medical knowledge, empathy, decision-making',
        'communication, patience, subject expertise',
        'network security, penetration testing, encryption',
        'deep learning, neural networks, Python',
        'mechanical design, thermodynamics, CAD',
        'market research, advertising, branding',
        'creativity, Photoshop, Illustrator',
        'structural analysis, project management, AutoCAD',
        'financial modeling, risk analysis, Excel',
        'counseling, behavioral analysis, empathy',
        'user research, wireframing, prototyping',
        'business strategy, leadership, agile methodologies',
        'laboratory research, clinical trials, diagnostics',
        'network protocols, troubleshooting, cloud computing',
        'writing, SEO, content strategy',
        'recruitment, employee relations, conflict resolution',
        'architectural design, 3D modeling, construction management',
        'business development, innovation, risk-taking'
    ],
    'personality_traits': [
        'analytical, logical', 'curious, mathematical', 'compassionate, decisive', 
        'patient, communicative', 'detail-oriented, strategic', 
        'innovative, problem-solving', 'practical, hands-on', 'persuasive, strategic',
        'creative, detail-oriented', 'structured, organized', 'analytical, detail-oriented',
        'empathetic, understanding', 'observant, user-focused', 'visionary, adaptable',
        'scientific, meticulous', 'logical, technical', 'articulate, expressive', 
        'social, people-oriented', 'artistic, detail-focused', 'risk-taking, independent'
    ],
    'education': [
        "CS Degree", "Data Science Master's", "Medical Degree", "Teaching Certificate",
        "Cybersecurity Certification", "AI Master's", "Mechanical Engineering Degree",
        "Marketing Degree", "Graphic Design Diploma", "Civil Engineering Degree",
        "Finance Degree", "Psychology Degree", "Design Degree", "MBA",
        "Biomedical Science Degree", "Networking Certification", "Journalism/English Degree",
        "HR Management Degree", "Architecture Degree", "Business Degree"
    ],
    'growth_outlook': [
        0.3, 0.4, 0.2, 0.1, 0.5, 
        0.6, 0.3, 0.4, 0.35, 0.3,
        0.4, 0.3, 0.5, 0.6, 0.35, 
        0.4, 0.3, 0.35, 0.4, 0.7
    ],
    'experience_levels': [
        'entry,mid,senior', 'entry,mid,senior', 'resident,specialist,consultant',
        'assistant,teacher,professor', 'junior,analyst,lead',
        'entry,mid,senior', 'entry,mid,senior', 'entry,mid,senior',
        'junior,designer,art director', 'entry,mid,senior',
        'analyst,manager,director', 'intern,therapist,clinical psychologist',
        'junior,mid,lead', 'associate,manager,senior manager',
        'junior,scientist,senior researcher', 'entry,mid,senior',
        'junior,writer,editor', 'entry,mid,senior',
        'intern,architect,senior architect', 'startup,scaleup,CEO'
    ]
})


# Vectorizer for Text Matching
vectorizer = TfidfVectorizer(stop_words='english')
career_vectors = vectorizer.fit_transform(career_data['required_skills'] + ' ' + career_data['personality_traits'])

# Function: Analyze Aspirations
def analyze_aspirations(text):
    doc = nlp(text)
    words = [token.lemma_ for token in doc if token.pos_ in ['NOUN', 'ADJ']]
    freq_dist = {word: words.count(word) for word in set(words)}
    total = sum(freq_dist.values())
    return {k: v/total for k, v in freq_dist.items() if v/total > 0.05}

# Function: Map Skills to Careers
def map_skills(user_skills):
    user_vec = vectorizer.transform([' '.join(user_skills)])
    similarities = cosine_similarity(user_vec, career_vectors)
    return pd.Series(similarities[0], index=career_data['career'])

# Function: Identify Skill Gaps
def identify_skill_gaps(career, user_skills):
    row = career_data[career_data['career'] == career].iloc[0]
    required_skills = row['required_skills'].split(', ')
    missing_skills = [s for s in required_skills if s not in user_skills]

    learning_resources = {
        'programming': ['LeetCode', 'Coursera CS'],
        'machine learning': ['Fast.ai', 'Coursera ML'],
        'data analysis': ['DataCamp', 'Google Analytics'],
        'communication': ['Toastmasters'],
    }
    
    recommendations = {skill: learning_resources.get(skill, ["General online courses"]) for skill in missing_skills}

    return {'missing_skills': missing_skills, 'learning_recommendations': recommendations}

# API Endpoint: Career Recommendation
@app.route('/recommend_career', methods=['POST'])
def recommend_career():
    data = request.json
    user_id = data['user_id']
    skills = data['skills']
    aspirations_text = data['aspirations']
    experience = data['experience']

    aspirations = analyze_aspirations(aspirations_text)
    skill_matches = map_skills(skills)

    aspiration_text = ' '.join([f"{k} {k}" * int(v*10) for k, v in aspirations.items()])
    aspiration_vec = vectorizer.transform([aspiration_text])
    aspiration_matches = pd.Series(cosine_similarity(aspiration_vec, career_vectors)[0], index=career_data['career'])

    combined_scores = (skill_matches + aspiration_matches) / 2

    experience_level = 'entry' if experience < 2 else 'mid' if experience < 5 else 'senior'
    valid_careers = [career for career in career_data['career'] if experience_level in career_data[career_data['career'] == career]['experience_levels'].iloc[0]]
    combined_scores = combined_scores[combined_scores.index.isin(valid_careers)]

    top_careers = combined_scores.sort_values(ascending=False).head(3)

    recommendations = []
    for career in top_careers.index:
        row = career_data[career_data['career'] == career].iloc[0]
        skill_gaps = identify_skill_gaps(career, skills)

        recommendations.append({
            'career': career,
            'match_score': round(top_careers[career] * 100, 1),
            'required_education': row['education'],
            'growth_outlook': row['growth_outlook'],
            'experience_path': row['experience_levels'].split(','),
            'skill_gaps': skill_gaps['missing_skills'],
            'learning_recommendations': skill_gaps['learning_recommendations']
        })

    return jsonify({'user_id': user_id, 'recommendations': recommendations})

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
