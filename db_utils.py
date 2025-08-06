import streamlit as st
import sqlite3
import pandas as pd

# Function to get a database connection
# Using st.connection for Streamlit's built-in connection management
def get_db_connection():
    return st.connection('turso', type='sql')

# Function to initialize the database and create tables if they don't exist
def init_db():
    conn = get_db_connection()
    with conn.session as s:
        s.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                has_children BOOLEAN,
                has_other_pets BOOLEAN,
                preferred_size TEXT,
                activity_level TEXT,
                preferred_pet TEXT
            );
        ''')
        s.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                report_type TEXT NOT NULL, -- 'lost', 'found', 'share'
                pet_name TEXT,
                pet_breed TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                points_earned INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
        ''')
        s.commit()

# Function to get a user by username
def get_user(username):
    conn = get_db_connection()
    user = conn.query('SELECT * FROM users WHERE username = ?', params=(username,), ttl=0)
    return user

# Function to create a new user
def create_user(username, password_hash, has_children, has_other_pets, preferred_size, activity_level, preferred_pet):
    conn = get_db_connection()
    with conn.session as s:
        s.execute(
            'INSERT INTO users (username, password, has_children, has_other_pets, preferred_size, activity_level, preferred_pet) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (username, password_hash, has_children, has_other_pets, preferred_size, activity_level, preferred_pet)
        )
        s.commit()

# Function to add a report and award points
def add_report(user_id, report_type, points, pet_name=None, pet_breed=None):
    conn = get_db_connection()
    with conn.session as s:
        s.execute(
            'INSERT INTO reports (user_id, report_type, pet_name, pet_breed, points_earned) VALUES (?, ?, ?, ?, ?)',
            (user_id, report_type, pet_name, pet_breed, points)
        )
        s.commit()

# Function to get a user's history
def get_user_history(user_id):
    conn = get_db_connection()
    history = conn.query('SELECT report_type, pet_name, pet_breed, timestamp, points_earned FROM reports WHERE user_id = ? ORDER BY timestamp DESC', params=(user_id,), ttl=0)
    return history

# Function to get the leaderboard
def get_leaderboard():
    conn = get_db_connection()
    leaderboard_df = conn.query('''
        SELECT u.username, SUM(r.points_earned) as total_points
        FROM reports r
        JOIN users u ON r.user_id = u.id
        GROUP BY u.username
        ORDER BY total_points DESC
        LIMIT 10;
    ''', ttl=60) # Cache for 60 seconds
    return leaderboard_df
