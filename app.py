import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter

# настройки страницы
st.set_page_config(page_title="Анализ винных данных", layout="wide")


# кэшируем загрузку данных
@st.cache_data
def load_data():
    df = pd.read_csv("data/wine_clean.csv")

    # Приведение типов
    for col in ["points", "price", "vintage", "alcohol", "price_per_point"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Стандартизация категорий (rose / rosé) ---
    if "category" in df.columns:
        df["category"] = (
            df["category"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        rose_map = {
            "rose": "rosé",
            "roze": "rosé",
            "rosè": "rosé",
            "rosê": "rosé",
            "rosė": "rosé",
            "rosé ": "rosé",
            "rosé": "rosé",
            "rosй": "rosé",
        }

        df["category"] = df["category"].replace(rose_map)

    return df



def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Фильтры в сайдбаре."""
    st.sidebar.header("Фильтры")

    # фильтр по стране
    if "country" in df.columns:
        countries = sorted(df["country"].dropna().unique().tolist())
        selected_countries = st.sidebar.multiselect(
            "Страны", options=countries, default=countries
        )
        if selected_countries:
            df = df[df["country"].isin(selected_countries)]

    # фильтр по категории
    if "category" in df.columns:
        categories = sorted(df["category"].dropna().unique().tolist())
        selected_categories = st.sidebar.multiselect(
            "Категории", options=categories, default=categories
        )
        if selected_categories:
            df = df[df["category"].isin(selected_categories)]

    # фильтр по цене
    if "price" in df.columns:
        min_price = float(df["price"].min())
        max_price = float(df["price"].max())
        price_range = st.sidebar.slider(
            "Цена, $",
            min_value=round(min_price),
            max_value=round(max_price),
            value=(round(min_price), round(max_price)),
            step=1,
        )
        df = df[(df["price"] >= price_range[0]) & (df["price"] <= price_range[1])]

    # фильтр по оценке
    if "points" in df.columns:
        min_points = int(df["points"].min())
        max_points = int(df["points"].max())
        points_range = st.sidebar.slider(
            "Оценка (points)",
            min_value=min_points,
            max_value=max_points,
            value=(min_points, max_points),
            step=1,
        )
        df = df[(df["points"] >= points_range[0]) & (df["points"] <= points_range[1])]

    return df


def plot_histogram(df, column, title, xlabel):
    """Простая гистограмма."""
    fig, ax = plt.subplots()
    ax.hist(df[column].dropna(), bins=30)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Частота")
    st.pyplot(fig)


def plot_scatter(df, x, y, hue=None, title=""):
    """Диаграмма рассеяния."""
    fig, ax = plt.subplots()
    if hue and hue in df.columns:
        sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax, alpha=0.7)
        ax.legend(title=hue, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        sns.scatterplot(data=df, x=x, y=y, ax=ax, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    st.pyplot(fig)


def plot_corr_heatmap(df, numeric_cols):
    """Тепловая карта корреляций."""
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Корреляция числовых признаков")
    st.pyplot(fig)


def get_top_words(text_series, top_n=20):
    """Подсчёт самых частых слов в описании."""
    all_text = " ".join(text_series.dropna().astype(str).tolist()).lower()
    tokens = re.findall(r"\b\w+\b", all_text)

    # короткий список стоп-слов (можно дополнять)
    stop_words = {
        "and", "the", "for", "with", "this", "that", "from", "but", "are",
        "was", "were", "you", "your", "his", "her", "she", "him", "its",
        "not", "all", "can", "will", "one", "two", "three", "wine"
    }
    tokens = [t for t in tokens if len(t) > 2 and t not in stop_words]

    counter = Counter(tokens)
    return counter.most_common(top_n)


def plot_top_words(top_words):
    """Горизонтальный bar-chart топ слов."""
    if not top_words:
        st.write("Недостаточно данных для анализа текста.")
        return
    words, counts = zip(*top_words)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(words, counts)
    ax.invert_yaxis()
    ax.set_xlabel("Частота")
    ax.set_title("Топ слов в описании")
    st.pyplot(fig)


def main():
    df = load_data()

    st.title("Анализ и визуализация данных о вине")
    st.caption("Данные: wine_clean (предобработанный набор)")

    # фильтрация
    filtered_df = filter_data(df)
    st.write(f"Показано строк: {len(filtered_df)} из {len(df)}")

    # вкладки
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Обзор данных", "Распределения", "Корреляции", "Текст описания"]
    )

    # --- Обзор данных ---
    with tab1:
        st.subheader("Таблица данных (первые 50 строк)")
        st.dataframe(filtered_df.head(50))

        st.subheader("Статистика числовых колонок")
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(filtered_df[numeric_cols].describe().T)
        else:
            st.write("Числовые колонки не найдены.")

    # --- Распределения ---
    with tab2:
        st.subheader("Распределения признаков")
        col1, col2 = st.columns(2)

        with col1:
            if "points" in filtered_df.columns:
                plot_histogram(filtered_df, "points", "Распределение оценок", "points")
            else:
                st.write("Колонка points отсутствует.")

        with col2:
            if "price" in filtered_df.columns:
                plot_histogram(filtered_df, "price", "Распределение цен", "Цена, $")
            else:
                st.write("Колонка price отсутствует.")

    # --- Корреляции ---
    with tab3:
        st.subheader("Корреляции числовых признаков")
        num_cols = [
            c
            for c in ["points", "price", "vintage", "alcohol", "price_per_point"]
            if c in filtered_df.columns
        ]
        if len(num_cols) >= 2:
            plot_corr_heatmap(filtered_df, num_cols)
        else:
            st.write("Недостаточно числовых признаков для тепловой карты.")

        st.subheader("Связь цены и оценки")
        if "price" in filtered_df.columns and "points" in filtered_df.columns:
            plot_scatter(
                filtered_df,
                x="price",
                y="points",
                hue="category" if "category" in filtered_df.columns else None,
                title="Цена vs Оценка",
            )
        else:
            st.write("Для scatter-графика нужны колонки price и points.")

    # --- Текст ---
    with tab4:
        st.subheader("Анализ текста описания (description)")
        if "description" in filtered_df.columns:
            top_words = get_top_words(filtered_df["description"], top_n=20)
            plot_top_words(top_words)
        else:
            st.write("Колонка description отсутствует в данных.")


if __name__ == "__main__":
    main()
