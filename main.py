import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Генерация данных о книгах


def generate_books():
    np.random.seed(42)
    books = pd.DataFrame({
        'book_id': range(1, 101),
        'title': [f'Book {i}' for i in range(1, 101)],
        'author': [f'Author {np.random.randint(1, 11)}' for _ in range(100)],
        'genre': np.random.choice(['Fiction', 'Non-fiction', 'Fantasy', 'Thriller', 'Mystery', 'Romance'], size=100),
        'year': np.random.randint(1950, 2023, size=100),
        'publisher': [f'Publisher {np.random.randint(1, 6)}' for _ in range(100)],
        'pages': np.random.randint(100, 1000, size=100),
        'rating': np.random.uniform(1, 5, size=100).round(2),
        'timestamp': pd.date_range(start='2000-01-01', periods=100)
    })
    return books

# Предобработка данных


def preprocess_data(books):
    # Можно добавить предобработку здесь, если необходимо
    return books

# Визуализация данных


def visualize_data(books):
    genre_counts = books['genre'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x='genre', data=books, ax=ax)
    ax.set_title('Распределение жанров книг')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# Кластеризация книг


def cluster_books(books):
    # Выбираем признаки для кластеризации
    features = ['year', 'rating']
    X = books[features]

    # Масштабируем признаки
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Применяем алгоритм кластеризации KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    books['cluster'] = kmeans.fit_predict(X_scaled)

    # Визуализируем результаты
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='year', y='rating', hue='cluster',
                    data=books, palette='viridis', ax=ax)
    ax.set_title('Кластеризация книг')
    st.pyplot(fig)


# Генерация данных о книгах
books = generate_books()
books = preprocess_data(books)

st.set_page_config(layout="wide")
st.title('Рекомендательная система для книг с аналитическими возможностями')

# Добавляем боковую панель для навигации
st.sidebar.title("Навигация")
analysis_type = st.sidebar.radio(
    "Выберите раздел анализа",
    ('Главная', 'Визуализация данных', 'Кластеризация книг',
     'Прогнозирование рейтинга', 'Выводы и интерпретации')
)

# На основе выбора показываем соответствующий раздел
if analysis_type == 'Главная':
    st.header('Добро пожаловать в систему рекомендаций книг!')
    st.markdown("...")

elif analysis_type == 'Визуализация данных':
    st.header('Визуализация данных')
    visualize_data(books)

elif analysis_type == 'Кластеризация книг':
    st.header('Кластеризация книг')
    cluster_books(books)

elif analysis_type == 'Прогнозирование рейтинга':
    st.header('Прогнозирование рейтинга книги')
    selected_book_title = st.selectbox('Выберите книгу', books['title'])
    selected_book_id = books[books['title'] ==
                             selected_book_title].iloc[0]['book_id']

    model = LinearRegression()
    model.fit(books[['book_id']], books['rating'])
    predicted_rating = model.predict([[selected_book_id]])[0]

    if st.button('Прогнозировать рейтинг для выбранной книги'):
        st.write(
            f'Прогнозируемый рейтинг для книги "{selected_book_title}": {predicted_rating:.2f}')

elif analysis_type == 'Выводы и интерпретации':
    st.header('Выводы и интерпретации')
    st.markdown("""
        - **Распределение жанров**: График показывает, какие жанры книг наиболее популярны среди выбранной выборки.
        - **Прогнозирование рейтинга**: Можно прогнозировать рейтинг для выбранной книги на основе имеющихся данных.
        """)
else:
    st.error("Выбран неверный раздел.")
