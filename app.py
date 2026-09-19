import streamlit as st
import joblib
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Configuración de página
st.set_page_config(
    page_title="Microproyecto 2",
    page_icon="ODS",
    layout="centered"
)

# Descarga stopwords
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')

download_nltk_resources()
stop_words_es = set(stopwords.words('spanish'))

# Función de preprocesamiento
def prepare_text(text):
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = SnowballStemmer(language='spanish')
    tokens = tokenizer.tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words_es]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# Diccionario de ODS
ODS_INFO = {
    1: {"nombre": "Fin de la pobreza", "color": "#E5243B"},
    2: {"nombre": "Hambre cero", "color": "#DDA63A"},
    3: {"nombre": "Salud y bienestar", "color": "#4C9F38"},
    4: {"nombre": "Educación de calidad", "color": "#C5192D"},
    5: {"nombre": "Igualdad de género", "color": "#FF3A21"},
    6: {"nombre": "Agua limpia y saneamiento", "color": "#26BDE2"},
    7: {"nombre": "Energía asequible y no contaminante", "color": "#FCC30B"},
    8: {"nombre": "Trabajo decente y crecimiento económico", "color": "#A21942"},
    9: {"nombre": "Industria, innovación e infraestructura", "color": "#FD6925"},
    10: {"nombre": "Reducción de las desigualdades", "color": "#DD1367"},
    11: {"nombre": "Ciudades y comunidades sostenibles", "color": "#FD9D24"},
    12: {"nombre": "Producción y consumo responsables", "color": "#BF8B2E"},
    13: {"nombre": "Acción por el clima", "color": "#3F7E44"},
    14: {"nombre": "Vida submarina", "color": "#0A97D9"},
    15: {"nombre": "Vida de ecosistemas terrestres", "color": "#56C02B"},
    16: {"nombre": "Paz, justicia e instituciones sólidas", "color": "#00689D"},
    17: {"nombre": "Alianzas para lograr los objetivos", "color": "#19486A"}
}

# Cargar modelo en memoria
@st.cache_resource
def load_model():
    return joblib.load("modelo_ods.joblib")

pipeline = load_model()

# Interfaz de usuario
st.title("Identificador de Relaciones Semánticas con ODS")
st.markdown(
    """
    Esta herramienta utiliza técnicas de **Procesamiento de Lenguaje Natural (PLN)** 
    y **Machine Learning (TF-IDF + LSA + LinearSVC)** para clasificar textos según 
    los **Objetivos de Desarrollo Sostenible (ODS)** de las Naciones Unidas.
    """
)

texto_usuario = st.text_area(
    "Ingresa un texto o fragmento a clasificar:",
    height=160,
    placeholder="Ej: Implementación de energías renovables como solar y eólica para reducir las emisiones de carbono..."
)

if st.button("Clasificar ODS", type="primary"):
    if not texto_usuario.strip():
        st.warning("Por favor, ingresa algún texto antes de clasificar.")
    else:
        with st.spinner("Analizando texto con el pipeline PLN..."):
            # Si el vectorizer ya tiene 'preprocessor=prepare_text', se puede pasar directo
            prediccion = pipeline.predict([texto_usuario])[0]
            ods_num = int(prediccion)
            info = ODS_INFO.get(ods_num, {"nombre": f"ODS {ods_num}", "color": "#333333"})

            st.success("¡Clasificación completada!")
            st.markdown(
                f"""
                <div style="padding: 18px; border-radius: 10px; background-color: {info['color']}; color: white; text-align: center;">
                    <h2 style="color: white; margin: 0;">ODS {ods_num}</h2>
                    <h4 style="color: white; margin: 5px 0 0 0;">{info['nombre']}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )
