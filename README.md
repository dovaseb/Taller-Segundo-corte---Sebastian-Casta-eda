# Taller-Segundo-corte---Sebastian-Casta-eda
Nombre: Sebastian Castañeda Salazar

---

### Ejercicio 1: Configuración del Entorno y Carga de Modelo Base

### Objetivo
Establecer el entorno de desarrollo necesario para trabajar con modelos LLM y cargar un modelo pre-entrenado utilizando las bibliotecas Transformers y PyTorch.

### Código
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# TODO: Configurar las variables de entorno para la caché de modelos
# Establecer la carpeta donde se almacenarán los modelos descargados
ruta_cache = './cache_modelos'
os.environ['TRANSFORMERS_CACHE'] = ruta_cache
os.makedirs(ruta_cache, exist_ok=True)
print(f"La caché de modelos se guardará en: {ruta_cache}")

def cargar_modelo(nombre_modelo):
    """
    Carga un modelo pre-entrenado y su tokenizador correspondiente.

    Args:
        nombre_modelo (str): Identificador del modelo en Hugging Face Hub

    Returns:
        tuple: (modelo, tokenizador)
    """
    # TODO: Implementar la carga del modelo y tokenizador
    # Utiliza AutoModelForCausalLM y AutoTokenizer
    try:
        tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
        modelo = AutoModelForCausalLM.from_pretrained(nombre_modelo)
        print(f"Modelo '{nombre_modelo}' y tokenizador cargados exitosamente.")
    except Exception as e:
        print(f"Error al cargar el modelo '{nombre_modelo}': {e}")
        return None, None

    # TODO: Configurar el modelo para inferencia (evaluar y usar half-precision si es posible)
    modelo.eval()
    if torch.cuda.is_available():
        modelo = modelo.half().cuda()
        print("Modelo cargado en la GPU y convertido a half-precision (float16).")
    else:
        print("Modelo cargado en la CPU.")

    return modelo, tokenizador

def verificar_dispositivo():
    """
    Verifica el dispositivo disponible (CPU/GPU) y muestra información relevante.

    Returns:
        torch.device: Dispositivo a utilizar
    """
    # TODO: Implementar la detección del dispositivo
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"CUDA versión: {torch.version.cuda}")
        print(f"Número de GPUs disponibles: {torch.cuda.device_count()}")
    else:
        dispositivo = torch.device("cpu")
        print("No se encontró GPU disponible, utilizando CPU.")

    return dispositivo

# Función principal de prueba
def main():
    dispositivo = verificar_dispositivo()
    print(f"Utilizando dispositivo: {dispositivo}")

    # TODO: Cargar un modelo pequeño adecuado para chatbots (ej. Mistral-7B, GPT2, etc.)
    nombre_modelo = "gpt2"  # Un modelo pequeño y rápido para pruebas
    modelo, tokenizador = cargar_modelo(nombre_modelo)

    if modelo is not None and tokenizador is not None:
        # TODO: Realizar una prueba simple de generación de texto
        texto_prompt = "Hola, ¿cómo estás?"
        input_ids = tokenizador.encode(texto_prompt, return_tensors="pt").to(dispositivo)

        with torch.no_grad():
            output = modelo.generate(input_ids, max_length=50, num_beams=5, no_repeat_ngram_size=2, temperature=1.0)

        texto_generado = tokenizador.decode(output[0], skip_special_tokens=True)
        print(f"\nPrompt: '{texto_prompt}'")
        print(f"Respuesta generada: '{texto_generado}'")

if __name__ == "__main__":
    main()
```
![image](https://github.com/user-attachments/assets/af513a77-a626-4c1b-8a1d-6f56de985884)

---

### Ejercicio 2: Procesamiento de Entrada y Generación de Respuestas

### Objetivo
Desarrollar las funciones necesarias para procesar la entrada del usuario, preparar los tokens para el modelo y generar respuestas coherentes.

### Código
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# (La configuración de la caché y la función cargar_modelo se mantienen del ejercicio anterior)
ruta_cache = './cache_modelos'
os.environ['TRANSFORMERS_CACHE'] = ruta_cache
os.makedirs(ruta_cache, exist_ok=True)
print(f"La caché de modelos se guardará en: {ruta_cache}")

def cargar_modelo(nombre_modelo):
    try:
        tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
        modelo = AutoModelForCausalLM.from_pretrained(nombre_modelo)
        print(f"Modelo '{nombre_modelo}' y tokenizador cargados exitosamente.")

        # Solución al error de padding para tokenizadores como GPT-2
        if tokenizador.pad_token is None:
            tokenizador.pad_token = tokenizador.eos_token
            print(f"Se estableció '{tokenizador.pad_token}' como pad_token.")

    except Exception as e:
        print(f"Error al cargar el modelo '{nombre_modelo}': {e}")
        return None, None

    modelo.eval()
    if torch.cuda.is_available():
        modelo = modelo.half().cuda()
        print("Modelo cargado en la GPU y convertido a half-precision (float16).")
    else:
        print("Modelo cargado en la CPU.")

    return modelo, tokenizador

def preprocesar_entrada(texto, tokenizador, longitud_maxima=512):
    """
    Preprocesa el texto de entrada para pasarlo al modelo.

    Args:
        texto (str): Texto de entrada del usuario
        tokenizador: Tokenizador del modelo
        longitud_maxima (int): Longitud máxima de la secuencia

    Returns:
        torch.Tensor: Tensor de entrada para el modelo
    """
    # TODO: Implementar el preprocesamiento
    # - Añadir tokens especiales si son necesarios (ej. [BOS], [SEP])
    # - Convertir a tensor
    # - Pasar al dispositivo correspondiente
    entrada_tokenizada = tokenizador.encode_plus(
        texto,
        add_special_tokens=True,
        max_length=longitud_maxima,
        padding='longest',
        truncation=True,
        return_tensors='pt'
    )
    entrada_procesada = entrada_tokenizada['input_ids'].to(modelo.device)  # Usamos el dispositivo del modelo
    return entrada_procesada

def generar_respuesta(modelo, entrada_procesada, tokenizador, parametros_generacion=None):
    """
    Genera una respuesta basada en la entrada procesada.

    Args:
        modelo: Modelo de lenguaje
        entrada_procesada: Tokens de entrada procesados
        tokenizador: Tokenizador del modelo
        parametros_generacion (dict): Parámetros para controlar la generación

    Returns:
        str: Respuesta generada
    """
    # TODO: Implementar valores por defecto para parámetros de generación
    if parametros_generacion is None:
        parametros_generacion = {
            'max_length': 100,
            'num_beams': 5,
            'no_repeat_ngram_size': 2,
            'temperature': 1.0,
            'top_p': 0.95
        }

    # TODO: Implementar la generación de texto
    # Utilizar modelo.generate() con los parámetros adecuados
    with torch.no_grad():
        output = modelo.generate(
            entrada_procesada,
            **parametros_generacion
        )

    # TODO: Decodificar la salida y limpiar la respuesta
    respuesta_tokenizada = output[:, entrada_procesada.shape[-1]:]  # Ignora el prompt en la salida
    respuesta = tokenizador.decode(respuesta_tokenizada[0], skip_special_tokens=True)

    return respuesta

def crear_prompt_sistema(instrucciones):
    """
    Crea un prompt de sistema para dar instrucciones al modelo.

    Args:
        instrucciones (str): Instrucciones sobre cómo debe comportarse el chatbot

    Returns:
        str: Prompt formateado
    """
    # TODO: Implementar la función para crear un prompt de sistema
    # Muchos modelos modernos no requieren un formato especial para el prompt del sistema
    # Se puede simplemente concatenar las instrucciones con la entrada del usuario.
    # Sin embargo, para modelos específicos, podría haber formatos como:
    # "System: {instrucciones}\n\nUser: {entrada}\n\nAssistant:"
    # Por ahora, devolvemos las instrucciones directamente para mayor flexibilidad.
    return instrucciones

# Ejemplo de uso
def interaccion_simple():
    nombre_modelo = "gpt2"  # Puedes cambiarlo a otro modelo
    modelo, tokenizador = cargar_modelo(nombre_modelo)

    if modelo is None or tokenizador is None:
        return

    # TODO: Crear un prompt de sistema para definir la personalidad del chatbot
    prompt_sistema = crear_prompt_sistema("Eres un chatbot amigable que responde preguntas de forma concisa.")

    while True:
        entrada_usuario = input("Usuario: ")
        if entrada_usuario.lower() == "salir":
            break

        # TODO: Procesar una entrada de ejemplo
        prompt_completo = f"{prompt_sistema} {entrada_usuario}"
        entrada_procesada = preprocesar_entrada(prompt_completo, tokenizador)

        # TODO: Generar y mostrar la respuesta
        respuesta = generar_respuesta(modelo, entrada_procesada, tokenizador)
        print(f"Chatbot: {respuesta}")

if __name__ == "__main__":
    interaccion_simple()
```
![image](https://github.com/user-attachments/assets/7fca89da-2212-422f-9aed-283ee535c39d)

---

### Ejercicio 3: Manejo de Contexto Conversacional

### Objetivo
Implementar un sistema para mantener el contexto de la conversación, permitiendo al chatbot recordar intercambios anteriores y responder coherentemente a conversaciones prolongadas.

### Código
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# (La configuración de la caché y la función cargar_modelo se mantienen)
ruta_cache = './cache_modelos'
os.environ['TRANSFORMERS_CACHE'] = ruta_cache
os.makedirs(ruta_cache, exist_ok=True)
print(f"La caché de modelos se guardará en: {ruta_cache}")

def cargar_modelo(nombre_modelo):
    try:
        tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
        modelo = AutoModelForCausalLM.from_pretrained(nombre_modelo)
        print(f"Modelo '{nombre_modelo}' y tokenizador cargados exitosamente.")
        if tokenizador.pad_token is None:
            tokenizador.pad_token = tokenizador.eos_token
            print(f"Se estableció '{tokenizador.pad_token}' como pad_token.")
    except Exception as e:
        print(f"Error al cargar el modelo '{nombre_modelo}': {e}")
        return None, None

    modelo.eval()
    if torch.cuda.is_available():
        modelo = modelo.half().cuda()
        print("Modelo cargado en la GPU y convertido a half-precision (float16).")
    else:
        print("Modelo cargado en la CPU.")

    return modelo, tokenizador

def verificar_dispositivo():
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"CUDA versión: {torch.version.cuda}")
        print(f"Número de GPUs disponibles: {torch.cuda.device_count()}")
    else:
        dispositivo = torch.device("cpu")
        print("No se encontró GPU disponible, utilizando CPU.")
    return dispositivo

def preprocesar_entrada(texto, tokenizador, longitud_maxima=512):
    entrada_tokenizada = tokenizador.encode_plus(
        texto,
        add_special_tokens=True,
        max_length=longitud_maxima,
        padding='longest',
        truncation=True,
        return_tensors='pt'
    )
    entrada_procesada = entrada_tokenizada['input_ids'].to(modelo.device)
    return entrada_procesada

def generar_respuesta(modelo, entrada_procesada, tokenizador, parametros_generacion=None):
    """
    Genera una respuesta basada en la entrada procesada.

    Args:
        modelo: Modelo de lenguaje
        entrada_procesada: Tokens de entrada procesados
        tokenizador: Tokenizador del modelo
        parametros_generacion (dict): Parámetros para controlar la generación

    Returns:
        str: Respuesta generada
    """
    if parametros_generacion is None:
        parametros_generacion = {
            'max_new_tokens': 100,  # Cambiado de 'max_length' a 'max_new_tokens'
            'num_beams': 5,
            'no_repeat_ngram_size': 2,
            'temperature': 1.0,
            'top_p': 0.95
        }
    with torch.no_grad():
        output = modelo.generate(
            entrada_procesada,
            **parametros_generacion
        )
    respuesta_tokenizada = output[:, entrada_procesada.shape[-1]:]
    respuesta = tokenizador.decode(respuesta_tokenizada[0], skip_special_tokens=True)
    return respuesta

class GestorContexto:
    """
    Clase para gestionar el contexto de una conversación con el chatbot.
    """

    def __init__(self, longitud_maxima=1024, formato_mensaje=None):
        """
        Inicializa el gestor de contexto.

        Args:
            longitud_maxima (int): Número máximo de tokens a mantener en el contexto
            formato_mensaje (callable): Función para formatear mensajes (por defecto, None)
        """
        self.historial = []
        self.longitud_maxima = longitud_maxima
        self.formato_mensaje = formato_mensaje or self._formato_predeterminado

    def _formato_predeterminado(self, rol, contenido):
        """
        Formato predeterminado para mensajes.

        Args:
            rol (str): 'sistema', 'usuario' o 'asistente'
            contenido (str): Contenido del mensaje

        Returns:
            str: Mensaje formateado
        """
        return f"{rol.capitalize()}: {contenido}"

    def agregar_mensaje(self, rol, contenido):
        """
        Agrega un mensaje al historial de conversación.

        Args:
            rol (str): 'sistema', 'usuario' o 'asistente'
            contenido (str): Contenido del mensaje
        """
        self.historial.append({"rol": rol, "contenido": contenido})

    def construir_prompt_completo(self):
        """
        Construye un prompt completo basado en el historial.

        Returns:
            str: Prompt completo para el modelo
        """
        mensajes_formateados = [self.formato_mensaje(msg["rol"], msg["contenido"]) for msg in self.historial]
        return "\n".join(mensajes_formateados)

    def truncar_historial(self, tokenizador):
        """
        Trunca el historial si excede la longitud máxima.

        Args:
            tokenizador: Tokenizador del modelo
        """
        tokens_actuales = 0
        historial_truncado = []

        # Mantener el primer mensaje si es del sistema
        if self.historial and self.historial[0]["rol"] == "sistema":
            historial_truncado.append(self.historial[0])
            tokens_actuales += len(tokenizador.encode(self.formato_mensaje(self.historial[0]["rol"], self.historial[0]["contenido"])))

        # Iterar en orden inverso para mantener los mensajes más recientes
        for mensaje in reversed(self.historial[1:] if self.historial and self.historial[0]["rol"] == "sistema" else self.historial):
            mensaje_formateado = self.formato_mensaje(mensaje["rol"], mensaje["contenido"])
            tokens_mensaje = len(tokenizador.encode(mensaje_formateado))
            if tokens_actuales + tokens_mensaje <= self.longitud_maxima:
                 # Insertar al principio si es el primer mensaje (después del sistema)
                 # O insertar después del mensaje del sistema si ya hay uno
                insert_index = 1 if self.historial and self.historial[0]["rol"] == "sistema" and len(historial_truncado) > 0 else 0
                historial_truncado.insert(insert_index, mensaje)
                tokens_actuales += tokens_mensaje
            else:
                break
        self.historial = historial_truncado


# Clase principal del chatbot
class Chatbot:
    """
    Implementación de chatbot con manejo de contexto.
    """

    def __init__(self, modelo_id, instrucciones_sistema=None):
        """
        Inicializa el chatbot.

        Args:
            modelo_id (str): Identificador del modelo en Hugging Face
            instrucciones_sistema (str): Instrucciones de comportamiento del sistema
        """
        self.modelo, self.tokenizador = cargar_modelo(modelo_id)
        self.dispositivo = verificar_dispositivo()
        # Ajustar la longitud máxima del contexto si es necesario, considerando el modelo utilizado.
        # GPT-2 tiene una longitud de contexto máxima de 1024 tokens.
        self.gestor_contexto = GestorContexto(longitud_maxima=self.tokenizador.model_max_length)


        # Inicializar el contexto con instrucciones del sistema
        if instrucciones_sistema:
            self.gestor_contexto.agregar_mensaje("sistema", instrucciones_sistema)

    def responder(self, mensaje_usuario, parametros_generacion=None):
        """
        Genera una respuesta al mensaje del usuario.

        Args:
            mensaje_usuario (str): Mensaje del usuario
            parametros_generacion (dict): Parámetros para la generación

        Returns:
            str: Respuesta del chatbot
        """
        # 1. Agregar mensaje del usuario al contexto
        self.gestor_contexto.agregar_mensaje("usuario", mensaje_usuario)

        # 2. Construir el prompt completo
        prompt_completo = self.gestor_contexto.construir_prompt_completo()

        # 3. Preprocesar la entrada (y truncar si es necesario antes de preprocesar)
        # Truncamos el historial ANTES de construir el prompt y preprocesar para asegurar que
        # el prompt_completo no exceda la longitud máxima manejable por el tokenizador/modelo.
        self.gestor_contexto.truncar_historial(self.tokenizador)
        prompt_completo_truncado = self.gestor_contexto.construir_prompt_completo()
        entrada_procesada = preprocesar_entrada(prompt_completo_truncado, self.tokenizador, longitud_maxima=self.tokenizador.model_max_length)


        # 4. Generar la respuesta
        respuesta = generar_respuesta(self.modelo, entrada_procesada, self.tokenizador, parametros_generacion)

        # 5. Agregar respuesta al contexto
        self.gestor_contexto.agregar_mensaje("asistente", respuesta)

        # 6. El truncado ya se realizó antes de generar la respuesta, si fue necesario.

        # 7. Devolver la respuesta
        return respuesta

# Prueba del sistema
def prueba_conversacion():
    # Crear una instancia del chatbot
    chatbot = Chatbot("gpt2", instrucciones_sistema="Eres un asistente útil y conciso.")

    # Simular una conversación de varios turnos
    print("Comienza la conversación (escribe 'salir' para terminar):")
    while True:
        mensaje_usuario = input("Usuario: ")
        if mensaje_usuario.lower() == "salir":
            break
        respuesta = chatbot.responder(mensaje_usuario)
        print(f"Asistente: {respuesta}")

if __name__ == "__main__":
    prueba_conversacion()
```
![image](https://github.com/user-attachments/assets/b9e38267-4acd-43c0-a53c-7c3c02e184a4)

---

### Ejercicio 4: Optimización del Modelo para Recursos Limitados

### Objetivo
Implementar técnicas de optimización para mejorar la velocidad de inferencia y reducir el consumo de memoria, permitiendo que el chatbot funcione eficientemente en dispositivos con recursos limitados.

### Código

```python
!pip install bitsandbytes
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
import time
import gc
import os

# --- Funciones de los ejercicios anteriores (para cargar el modelo) ---
ruta_cache = './cache_modelos'
os.environ['TRANSFORMERS_CACHE'] = ruta_cache
os.makedirs(ruta_cache, exist_ok=True)
print(f"La caché de modelos se guardará en: {ruta_cache}")

def verificar_dispositivo():
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"CUDA versión: {torch.version.cuda}")
        print(f"Número de GPUs disponibles: {torch.cuda.device_count()}")
    else:
        dispositivo = torch.device("cpu")
        print("No se encontró GPU disponible, utilizando CPU.")
    return dispositivo

def cargar_modelo(nombre_modelo):
    try:
        tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
        modelo = AutoModelForCausalLM.from_pretrained(nombre_modelo)
        print(f"Modelo '{nombre_modelo}' y tokenizador cargados exitosamente.")
        if tokenizador.pad_token is None:
            tokenizador.pad_token = tokenizador.eos_token
            print(f"Se estableció '{tokenizador.pad_token}' como pad_token.")
    except Exception as e:
        print(f"Error al cargar el modelo '{nombre_modelo}': {e}")
        return None, None

    modelo.eval()
    if torch.cuda.is_available():
        modelo = modelo.half().cuda()
        print("Modelo cargado en la GPU y convertido a half-precision (float16).")
    else:
        print("Modelo cargado en la CPU.")

    return modelo, tokenizador

# --- Funciones para la optimización (Ejercicio 4) ---
def configurar_cuantizacion(bits=4, double_quant=True, llm_int8_threshold=6.0, quant_type='nf4'):
    """
    Configura los parámetros para la cuantización del modelo.

    Args:
        bits (int): Bits para cuantización (4 u 8)
        double_quant (bool): Usar doble cuantización para mayor ahorro de memoria (solo para 4 bits)
        llm_int8_threshold (float): Umbral para la cuantización int8 (para modelos que lo soportan)
        quant_type (str): Tipo de cuantización ('nf4' o 'bnb4' para 4 bits, 'int8' para 8 bits)

    Returns:
        BitsAndBytesConfig: Configuración de cuantización
    """
    if bits not in [4, 8]:
        raise ValueError("Los bits deben ser 4 u 8.")

    if bits == 4:
        config_cuantizacion = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=torch.float16  # Recomendado para rendimiento
        )
    elif bits == 8:
        config_cuantizacion = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=llm_int8_threshold
        )
    return config_cuantizacion

def cargar_modelo_optimizado(nombre_modelo, optimizaciones=None):
    """
    Carga un modelo con optimizaciones aplicadas.

    Args:
        nombre_modelo (str): Identificador del modelo
        optimizaciones (dict): Diccionario con flags para las optimizaciones

    Returns:
        tuple: (modelo, tokenizador)
    """
    tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
    config = AutoConfig.from_pretrained(nombre_modelo)
    load_in_8bit = False
    load_in_4bit = False
    quantization_config = None

    if optimizaciones is None:
        optimizaciones = {
            "cuantizacion": False,
            "bits": 4,
            "double_quant": True,
            "llm_int8_threshold": 6.0,
            "quant_type": 'nf4',
            "offload_cpu": False,
            "flash_attention": False, # Flash Attention 2 se habilita directamente en la config si es compatible
            "sliding_window": None # Tamaño de la ventana deslizante
        }

    if optimizaciones.get("cuantizacion", False):
        if optimizaciones.get("bits") == 4:
            load_in_4bit = True
            quantization_config = configurar_cuantizacion(
                bits=4,
                double_quant=optimizaciones.get("double_quant", True),
                quant_type=optimizaciones.get("quant_type", 'nf4')
            )
        elif optimizaciones.get("bits") == 8:
            load_in_8bit = True
            quantization_config = configurar_cuantizacion(
                bits=8,
                llm_int8_threshold=optimizaciones.get("llm_int8_threshold", 6.0)
            )

    try:
        modelo = AutoModelForCausalLM.from_pretrained(
            nombre_modelo,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() and (load_in_4bit or load_in_8bit) else None,
            device_map="auto" if optimizaciones.get("offload_cpu") else None,
            attn_implementation="flash_attention_2" if optimizaciones.get("flash_attention") and config.attn_implementation == "flash_attention_2" else "eager"
        )
        print(f"Modelo '{nombre_modelo}' cargado con optimizaciones: {optimizaciones}")
    except Exception as e:
        print(f"Error al cargar el modelo '{nombre_modelo}' con optimizaciones: {e}")
        modelo = AutoModelForCausalLM.from_pretrained(nombre_modelo, torch_dtype=torch.float16 if torch.cuda.is_available() else None, device_map="auto")

    return modelo, tokenizador

def aplicar_sliding_window(modelo, window_size=1024):
    """
    Configura la atención de ventana deslizante para procesar secuencias largas.

    Args:
        modelo: Modelo a configurar
        window_size (int): Tamaño de la ventana de atención
    """
    config = modelo.config
    if hasattr(config, "attn_config") and hasattr(config.attn_config, "sliding_window"):
        config.attn_config.sliding_window = window_size
        modelo.config = config
        print(f"Atención de ventana deslizante configurada con tamaño: {window_size}")
    elif hasattr(config, "sliding_window"):
        config.sliding_window = window_size
        modelo.config = config
        print(f"Atención de ventana deslizante configurada con tamaño: {window_size}")
    else:
        print(f"El modelo {modelo.__class__.__name__} no soporta directamente la atención de ventana deslizante.")

def evaluar_rendimiento(modelo, tokenizador, texto_prueba, dispositivo="cuda" if torch.cuda.is_available() else "cpu", num_runs=5):
    """
    Evalúa el rendimiento del modelo en términos de velocidad y memoria.

    Args:
        modelo: Modelo a evaluar
        tokenizador: Tokenizador del modelo
        texto_prueba (str): Texto para pruebas de rendimiento
        dispositivo: Dispositivo donde se ejecutará
        num_runs (int): Número de veces para ejecutar la inferencia y calcular el promedio

    Returns:
        dict: Métricas de rendimiento
    """
    modelo.to(dispositivo)
    modelo.eval()

    input_ids = tokenizador.encode(texto_prueba, return_tensors="pt").to(dispositivo)
    input_length = input_ids.shape[-1]

    times = []
    memory_usage = []
    for _ in range(num_runs):
        torch.cuda.empty_cache()
        gc.collect()
        start_time = time.time()
        with torch.no_grad():
            _ = modelo.generate(input_ids, max_length=input_length + 50)
        end_time = time.time()
        times.append(end_time - start_time)
        memory_usage.append(torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0)

    avg_time = sum(times) / num_runs
    avg_memory = sum(memory_usage) / num_runs
    tokens_per_second = (input_length * num_runs) / sum(times) if sum(times) > 0 else 0

    metricas = {
        "inference_time_avg (seconds)": f"{avg_time:.4f}",
        "memory_usage_max_avg (MB)": f"{avg_memory:.2f}",
        "tokens_per_second": f"{tokens_per_second:.2f}",
        "device": dispositivo
    }
    return metricas

# Función de demostración
def demo_optimizaciones(nombre_modelo="gpt2", texto_prueba="La inteligencia artificial es"):
    """
    Crea y evalúa diferentes configuraciones del modelo para comparar el rendimiento.
    """
    print(f"Evaluando el modelo: {nombre_modelo}")
    dispositivo = verificar_dispositivo()

    print("\n--- Modelo Base (float16) ---")
    modelo_base, tokenizador_base = cargar_modelo(nombre_modelo)
    if modelo_base:
        metrics_base = evaluar_rendimiento(modelo_base, tokenizador_base, texto_prueba, dispositivo)
        print(metrics_base)
        del modelo_base
        del tokenizador_base
        torch.cuda.empty_cache()
        gc.collect()

    print("\n--- Modelo con Cuantización 4 bits (nf4, double_quant=True) ---")
    modelo_4bit, tokenizador_4bit = cargar_modelo_optimizado(nombre_modelo, {"cuantizacion": True, "bits": 4, "double_quant": True, "quant_type": 'nf4'})
    if modelo_4bit:
        metrics_4bit = evaluar_rendimiento(modelo_4bit, tokenizador_4bit, texto_prueba, dispositivo)
        print(metrics_4bit)
        del modelo_4bit
        del tokenizador_4bit
        torch.cuda.empty_cache()
        gc.collect()

    # Verificar si el modelo soporta sliding window attention
    config = AutoConfig.from_pretrained(nombre_modelo)
    supports_sliding_window = hasattr(config, "attn_config") and hasattr(config.attn_config, "sliding_window") or hasattr(config, "sliding_window")

    if supports_sliding_window:
        print("\n--- Modelo con Sliding Window Attention (window_size=512) ---")
        modelo_sw, tokenizador_sw = cargar_modelo(nombre_modelo)
        if modelo_sw:
            aplicar_sliding_window(modelo_sw, window_size=512)
            metrics_sw = evaluar_rendimiento(modelo_sw, tokenizador_sw, texto_prueba, dispositivo)
            print(metrics_sw)
            del modelo_sw
            del tokenizador_sw
            torch.cuda.empty_cache()
            gc.collect()
    else:
        print(f"\nEl modelo {nombre_modelo} no soporta directamente Sliding Window Attention.")

    print("\n--- Modelo con Cuantización 4 bits y (si es compatible) Flash Attention ---")
    optims_all = {"cuantizacion": True, "bits": 4, "double_quant": True, "quant_type": 'nf4', "flash_attention": True}
    modelo_all, tokenizador_all = cargar_modelo_optimizado(nombre_modelo, optims_all)
    if modelo_all:
        metrics_all = evaluar_rendimiento(modelo_all, tokenizador_all, texto_prueba, dispositivo)
        print(metrics_all)
        del modelo_all
        del tokenizador_all
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    demo_optimizaciones()
```
![image](https://github.com/user-attachments/assets/9eac1777-35fc-4cf3-a5f5-a54b3c4dce0e)
![image](https://github.com/user-attachments/assets/a8d32857-39e0-441e-9c89-aa8bc9eadc81)

---

### Ejercicio 5: Personalización del Chatbot y Despliegue

### Objetivo
Implementar técnicas para personalizar el comportamiento del chatbot y prepararlo para su despliegue como una aplicación web simple.

### Código 

```python
pip install gradio peft transformers torch
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
import os
import gradio as gr
from peft import LoraConfig, get_peft_model, TaskType, PeftModel 
import gc
import time

# --- Configuración de Caché y Dispositivo (de ejercicios anteriores) ---
ruta_cache = './cache_modelos'
os.environ['TRANSFORMERS_CACHE'] = ruta_cache
os.makedirs(ruta_cache, exist_ok=True)
print(f"La caché de modelos se guardará en: {ruta_cache}")

# Variable global para el modelo y tokenizador (simplifica el acceso en Gradio)
# Se inicializarán en main_despliegue o al cargar un modelo personalizado
modelo_global = None
tokenizador_global = None
dispositivo_global = None

def verificar_dispositivo():
    """
    Verifica el dispositivo disponible (CPU/GPU) y muestra información relevante.
    """
    global dispositivo_global
    if torch.cuda.is_available():
        dispositivo_global = torch.device("cuda")
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"CUDA versión: {torch.version.cuda}")
        print(f"Número de GPUs disponibles: {torch.cuda.device_count()}")
    else:
        dispositivo_global = torch.device("cpu")
        print("No se encontró GPU disponible, utilizando CPU.")
    return dispositivo_global

def cargar_modelo_base(nombre_modelo, optimizaciones=None):
    """
    Carga un modelo base, opcionalmente con optimizaciones como cuantización.
    Similar a cargar_modelo_optimizado del Codigo 4.
    """
    global modelo_global, tokenizador_global, dispositivo_global
    if dispositivo_global is None:
        dispositivo_global = verificar_dispositivo()

    tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
    config = AutoConfig.from_pretrained(nombre_modelo)

    if tokenizador.pad_token is None:
        tokenizador.pad_token = tokenizador.eos_token
        print(f"Se estableció '{tokenizador.pad_token}' como pad_token para el tokenizador.")

    load_in_8bit = False
    load_in_4bit = False
    quantization_config_bnb = None # Renombrado para evitar conflicto

    if optimizaciones is None:
        optimizaciones = {
            "cuantizacion": False, # Por defecto no cuantizar al cargar modelo base para fine-tuning
        }

    if optimizaciones.get("cuantizacion", False) and dispositivo_global.type == 'cuda':
        bits = optimizaciones.get("bits", 4) # Default a 4 bits si cuantización está activa
        if bits == 4:
            load_in_4bit = True
            quantization_config_bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=optimizaciones.get("double_quant", True),
                bnb_4bit_quant_type=optimizaciones.get("quant_type", 'nf4'),
                bnb_4bit_compute_dtype=torch.float16
            )
            print("Configurando cuantización de 4 bits.")
        elif bits == 8:
            load_in_8bit = True
            quantization_config_bnb = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=optimizaciones.get("llm_int8_threshold", 6.0)
            )
            print("Configurando cuantización de 8 bits.")
    else:
        print("Cuantización no habilitada o no hay GPU disponible. Cargando en float16/float32.")


    try:
        modelo = AutoModelForCausalLM.from_pretrained(
            nombre_modelo,
            quantization_config=quantization_config_bnb if (load_in_4bit or load_in_8bit) else None,
            torch_dtype=torch.float16 if dispositivo_global.type == 'cuda' and not (load_in_4bit or load_in_8bit) else (torch.bfloat16 if load_in_4bit or load_in_8bit else None), # bfloat16 para cómputo con cuantización
            device_map="auto" if dispositivo_global.type == 'cuda' else None, # device_map="auto" para múltiples GPUs o offloading
            attn_implementation="flash_attention_2" if optimizaciones.get("flash_attention", False) and hasattr(config, "attn_implementation") and config.attn_implementation == "flash_attention_2" and dispositivo_global.type == 'cuda' else "eager",
        )
        print(f"Modelo '{nombre_modelo}' cargado exitosamente.")
        if not (load_in_4bit or load_in_8bit) and dispositivo_global.type == 'cpu':
             print("Modelo cargado en CPU en precisión completa (float32).")
        elif not (load_in_4bit or load_in_8bit) and dispositivo_global.type == 'cuda':
            print("Modelo cargado en GPU en half-precision (float16).")


    except Exception as e:
        print(f"Error al cargar el modelo '{nombre_modelo}' con optimizaciones: {e}")
        print("Intentando cargar el modelo sin optimizaciones específicas de cuantización/dtype en CPU.")
        modelo = AutoModelForCausalLM.from_pretrained(nombre_modelo) # Fallback más simple
        if dispositivo_global.type == 'cuda':
            modelo = modelo.to(dispositivo_global)


    modelo_global = modelo
    tokenizador_global = tokenizador
    modelo_global.eval() # Por defecto en modo evaluación
    return modelo_global, tokenizador_global

def preprocesar_entrada(texto, tokenizador, longitud_maxima=512):
    global modelo_global # Necesitamos acceso al dispositivo del modelo
    if modelo_global is None:
        raise ValueError("El modelo global no ha sido cargado.")
    if tokenizador is None:
        raise ValueError("El tokenizador global no ha sido cargado.")

    entrada_tokenizada = tokenizador.encode_plus(
        texto,
        add_special_tokens=True,
        max_length=longitud_maxima,
        padding='longest',
        truncation=True,
        return_tensors='pt'
    )
    # Asegúrate de que el dispositivo del modelo_global sea el correcto
    entrada_procesada = entrada_tokenizada['input_ids'].to(modelo_global.device)
    return entrada_procesada

def generar_respuesta(modelo, entrada_procesada, tokenizador, parametros_generacion=None):
    if parametros_generacion is None:
        parametros_generacion = {
            'max_new_tokens': 100,
            'num_beams': 3, # Reducido para posible ejecución en CPU
            'no_repeat_ngram_size': 2,
            'temperature': 0.8, # Ligeramente menos aleatorio
            'top_p': 0.92,
            'pad_token_id': tokenizador.eos_token_id # Importante para generación correcta
        }
    with torch.no_grad():
        output = modelo.generate(
            entrada_procesada,
            **parametros_generacion
        )
    # Decodificar solo los tokens nuevos generados
    respuesta_tokenizada = output[:, entrada_procesada.shape[-1]:]
    respuesta = tokenizador.decode(respuesta_tokenizada[0], skip_special_tokens=True)
    return respuesta.strip()

class GestorContexto:
    def __init__(self, longitud_maxima=1024, formato_mensaje=None):
        self.historial = []
        self.longitud_maxima = longitud_maxima # En tokens
        self.formato_mensaje = formato_mensaje or self._formato_predeterminado

    def _formato_predeterminado(self, rol, contenido):
        return f"{rol.capitalize()}: {contenido}"

    def agregar_mensaje(self, rol, contenido):
        self.historial.append({"rol": rol, "contenido": contenido})

    def construir_prompt_completo(self):
        mensajes_formateados = [self.formato_mensaje(msg["rol"], msg["contenido"]) for msg in self.historial]
        return "\n".join(mensajes_formateados)

    def truncar_historial(self, tokenizador):
        global tokenizador_global
        if tokenizador is None: # Usa el global si no se pasa uno específico
            tokenizador = tokenizador_global
        if tokenizador is None:
            print("Advertencia: No hay tokenizador para truncar historial.")
            return

        tokens_totales_aproximados = 0
        # Primero calcular tokens de todo el historial formateado
        prompt_actual = self.construir_prompt_completo()
        tokens_actuales = len(tokenizador.encode(prompt_actual))

        while tokens_actuales > self.longitud_maxima and len(self.historial) > 1:
            # Priorizar eliminar mensajes antiguos, excepto el de sistema si existe
            if self.historial[0]["rol"] == "sistema" and len(self.historial) > 1:
                del self.historial[1] # Elimina el mensaje más antiguo después del de sistema
            else:
                del self.historial[0] # Elimina el mensaje más antiguo
            
            prompt_actual = self.construir_prompt_completo() # Reconstruir para recalcular
            tokens_actuales = len(tokenizador.encode(prompt_actual))

        if tokens_actuales > self.longitud_maxima and len(self.historial) == 1:
             # Si incluso un solo mensaje (o el de sistema) es demasiado largo, truncarlo
            print("Advertencia: El mensaje restante es demasiado largo, se truncará su contenido.")
            mensaje_unico = self.historial[0]
            ids_truncados = tokenizador.encode(mensaje_unico["contenido"], max_length=self.longitud_maxima, truncation=True)
            self.historial[0]["contenido"] = tokenizador.decode(ids_truncados, skip_special_tokens=True)


class Chatbot:
    def __init__(self, modelo_id_o_path, instrucciones_sistema=None, es_personalizado=False, optimizaciones_carga=None):
        global modelo_global, tokenizador_global, dispositivo_global
        self.dispositivo = verificar_dispositivo() # Asegura que dispositivo_global esté seteado

        if es_personalizado:
            print(f"Cargando modelo PEFT personalizado desde: {modelo_id_o_path}")
            self.modelo, self.tokenizador = cargar_modelo_personalizado(modelo_id_o_path)
        else:
            print(f"Cargando modelo base: {modelo_id_o_path}")
            self.modelo, self.tokenizador = cargar_modelo_base(modelo_id_o_path, optimizaciones=optimizaciones_carga)
        
        modelo_global = self.modelo # Actualiza globales
        tokenizador_global = self.tokenizador

        # Determinar longitud máxima del contexto del modelo si está disponible
        model_max_len = getattr(self.tokenizador, 'model_max_length', 1024)
        if hasattr(self.modelo, 'config') and hasattr(self.modelo.config, 'max_position_embeddings'):
            model_max_len = self.modelo.config.max_position_embeddings
        
        self.gestor_contexto = GestorContexto(longitud_maxima=model_max_len // 2) # Usar la mitad para dar espacio a la respuesta

        if instrucciones_sistema:
            self.gestor_contexto.agregar_mensaje("sistema", instrucciones_sistema)

    def responder(self, mensaje_usuario, parametros_generacion=None):
        if self.modelo is None or self.tokenizador is None:
            return "Error: El modelo o el tokenizador no están cargados."

        self.gestor_contexto.agregar_mensaje("usuario", mensaje_usuario)
        self.gestor_contexto.truncar_historial(self.tokenizador) # Truncar antes de construir prompt

        prompt_completo = self.gestor_contexto.construir_prompt_completo()
        # Añadir un sufijo que indique al modelo que debe generar la respuesta del asistente
        # Esto es crucial para muchos modelos instructivos o de chat.
        if not prompt_completo.endswith("Asistente:"): # Evitar duplicados si ya está
             prompt_completo += "\nAsistente:"


        # Usar model_max_length del tokenizador para preprocesar_entrada
        max_len_tokenizador = getattr(self.tokenizador, 'model_max_length', 1024)

        entrada_procesada = preprocesar_entrada(prompt_completo, self.tokenizador, longitud_maxima=max_len_tokenizador)

        respuesta_chatbot = generar_respuesta(self.modelo, entrada_procesada, self.tokenizador, parametros_generacion)

        self.gestor_contexto.agregar_mensaje("asistente", respuesta_chatbot)
        # No es necesario truncar de nuevo aquí si se hizo bien antes.

        return respuesta_chatbot

# --- Funciones del Ejercicio 5 ---

def configurar_peft(modelo, r=8, lora_alpha=32, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM, target_modules=None):
    """
    Configura el modelo para fine-tuning con PEFT/LoRA.

    Args:
        modelo: Modelo base a adaptar.
        r (int): Rango de adaptadores LoRA (dimensión de las matrices de bajo rango).
        lora_alpha (int): Escala alpha para LoRA. Es un hiperparámetro que escala los pesos aprendidos.
                         Una práctica común es poner lora_alpha = 2 * r.
        lora_dropout (float): Dropout para las capas LoRA.
        task_type: Tipo de tarea para PEFT (ej. CAUSAL_LM).
        target_modules (list of str, opcional): Nombres de los módulos a los que aplicar LoRA.
                            Si es None, PEFT intentará inferirlos (puede no ser óptimo para todos los modelos).
                            Ejemplos: ["q_proj", "v_proj"] para muchos transformadores.
                            Para GPT2: ["c_attn"] o ser más específico como ["c_attn.q_proj", "c_attn.v_proj"] si c_attn es un nn.Linear grande.
                            Para modelos como Llama: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    Returns:
        modelo_peft: Modelo adaptado para fine-tuning.
    """
    # TODO: Implementar la configuración de PEFT
    if target_modules is None:
        # Intento genérico para modelos tipo GPT-2/GPT-J/NeoX
        print("Advertencia: target_modules no especificado. PEFT intentará inferir.")
        pass


    config_lora = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules, # Dejar que PEFT infiera si es None y el modelo lo soporta
        lora_dropout=lora_dropout,
        bias="none",  # 'none', 'all', or 'lora_only'. 'none' es común.
        task_type=task_type
    )


    # Si el modelo ya está cuantizado (ej. load_in_4bit/8bit), PEFT necesita preparación especial
    if getattr(modelo, "is_loaded_in_8bit", False) or getattr(modelo, "is_loaded_in_4bit", False):
        from peft import prepare_model_for_kbit_training
        print("Preparando modelo cuantizado para entrenamiento con PEFT...")
        modelo = prepare_model_for_kbit_training(
            modelo, use_gradient_checkpointing=True # GC es muy recomendado aquí
        )

    try:
        modelo_peft = get_peft_model(modelo, config_lora)
        print("Modelo configurado con PEFT/LoRA.")
        modelo_peft.print_trainable_parameters()
    except Exception as e:
        print(f"Error al aplicar get_peft_model: {e}")
        print("Asegúrate de que 'target_modules' sea compatible con la arquitectura del modelo si PEFT no puede inferirlos.")
        return modelo # Devuelve el modelo original si falla

    return modelo_peft

def guardar_modelo_peft(modelo_peft, tokenizador, ruta_base, nombre_adaptador="lora_chatbot"):
    """
    Guarda los adaptadores PEFT (LoRA) y el tokenizador.
    El modelo base no se guarda aquí, solo los adaptadores.

    Args:
        modelo_peft: Modelo con adaptadores PEFT.
        tokenizador: Tokenizador del modelo.
        ruta_base (str): Directorio base donde se creará una subcarpeta para el adaptador.
        nombre_adaptador (str): Nombre de la subcarpeta para el adaptador.
    """
    ruta_adaptador = os.path.join(ruta_base, nombre_adaptador)
    os.makedirs(ruta_adaptador, exist_ok=True)
    try:
        modelo_peft.save_pretrained(ruta_adaptador)
        if tokenizador is not None:
            tokenizador.save_pretrained(ruta_adaptador)
        print(f"Adaptador PEFT y tokenizador guardados en: {ruta_adaptador}")
    except Exception as e:
        print(f"Error al guardar el modelo PEFT o el tokenizador: {e}")

def cargar_modelo_personalizado(ruta_adaptador, nombre_modelo_base=None):
    """
    Carga un modelo base y luego aplica los adaptadores PEFT desde una ruta específica.

    Args:
        ruta_adaptador (str): Ruta donde se guardaron los adaptadores PEFT (y el tokenizador).
        nombre_modelo_base (str, opcional): Nombre o ruta del modelo base original.
                                         PEFT >=0.7.0 puede inferirlo si se guardó con el adaptador.
                                         Si no, debe proporcionarse.

    Returns:
        tuple: (modelo_peft, tokenizador) o (None, None) si falla.
    """
    global modelo_global, tokenizador_global, dispositivo_global
    if dispositivo_global is None:
        dispositivo_global = verificar_dispositivo()

    try:
        # Cargar tokenizador
        tokenizador = AutoTokenizer.from_pretrained(ruta_adaptador)
        if tokenizador.pad_token is None:
            tokenizador.pad_token = tokenizador.eos_token

        # Configuración para cargar el modelo base (puede ser cuantizado)
        # Intentar leer la configuración del modelo base desde el adaptador PEFT si existe
        peft_config = PeftConfig.from_pretrained(ruta_adaptador)
        nombre_modelo_base_resolved = nombre_modelo_base or peft_config.base_model_name_or_path
        
        print(f"Cargando modelo base '{nombre_modelo_base_resolved}' para aplicar adaptadores PEFT.")
        bnb_config_base = None

        modelo_base = AutoModelForCausalLM.from_pretrained(
            nombre_modelo_base_resolved,
            quantization_config=bnb_config_base, # Aplicar si el base fue cuantizado
            torch_dtype=torch.float16 if dispositivo_global.type == 'cuda' and bnb_config_base is None else None,
            device_map="auto" if dispositivo_global.type == 'cuda' else None,
        )
        print(f"Modelo base '{nombre_modelo_base_resolved}' cargado.")

        # Aplicar adaptadores PEFT
        modelo_peft = PeftModel.from_pretrained(modelo_base, ruta_adaptador)
        modelo_peft = modelo_peft.merge_and_unload() # Opcional: fusionar para inferencia más rápida si no se va a entrenar más
                                                    # Esto crea un nuevo modelo estándar. Si se quiere seguir entrenando, omitir.
                                                    # Para inferencia, fusionar es bueno.
        print(f"Adaptadores PEFT de '{ruta_adaptador}' aplicados al modelo base.")
        
        modelo_peft.eval()
        if dispositivo_global.type == 'cuda' and bnb_config_base is None : # Si no está cuantizado y hay GPU
             if not hasattr(modelo_peft, 'hf_device_map'): # si no tiene device_map (no multi-gpu/offload)
                modelo_peft = modelo_peft.to(dispositivo_global)

        modelo_global = modelo_peft
        tokenizador_global = tokenizador
        return modelo_peft, tokenizador

    except Exception as e:
        print(f"Error al cargar el modelo PEFT personalizado: {e}")
        print("Asegúrate de que 'ruta_adaptador' sea correcta y que 'nombre_modelo_base' (si es necesario) también lo sea.")
        return None, None


# --- Interfaz Web con Gradio ---
chatbot_instance = None # Variable global para la instancia del chatbot

def inicializar_chatbot_global(modelo_id_o_path, instrucciones_sistema, es_personalizado, nombre_modelo_base_peft=None):
    """Inicializa o reinicializa la instancia global del chatbot."""
    global chatbot_instance, modelo_global, tokenizador_global
    print("Limpiando instancias previas del modelo y chatbot...")
    if modelo_global:
        del modelo_global
        modelo_global = None
    if tokenizador_global:
        del tokenizador_global
        tokenizador_global = None
    if chatbot_instance:
        del chatbot_instance
        chatbot_instance = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Inicializando chatbot con: {modelo_id_o_path}")
    if es_personalizado:
        chatbot_instance = Chatbot(modelo_id_o_path,
                                   instrucciones_sistema=instrucciones_sistema,
                                   es_personalizado=True) # El constructor de Chatbot llamará a cargar_modelo_personalizado
    else:
        chatbot_instance = Chatbot(modelo_id_o_path,
                                   instrucciones_sistema=instrucciones_sistema,
                                   es_personalizado=False,
                                   optimizaciones_carga={"cuantizacion": True, "bits": 4} if dispositivo_global.type == 'cuda' else None) # Cargar con cuantización si hay GPU
    return "Chatbot inicializado."

def chatbot_respuesta_gradio(mensaje_usuario, historial_chat):
    """
    Función de callback para Gradio.
    historial_chat es una lista de tuplas [(user_msg1, bot_msg1), (user_msg2, bot_msg2), ...]
    """
    global chatbot_instance
    if chatbot_instance is None:
        print("Advertencia: Chatbot no inicializado. Intentando inicialización por defecto.")
        return "Error: El chatbot no está inicializado. Por favor, carga un modelo.", historial_chat

    print(f"Usuario (Gradio): {mensaje_usuario}")
    respuesta = chatbot_instance.responder(mensaje_usuario)
    print(f"Chatbot (Gradio): {respuesta}")
    return respuesta # Gradio ChatInterface espera solo la respuesta del bot

def crear_interfaz_web():
    """
    Crea una interfaz web para el chatbot usando Gradio.
    """
    global chatbot_instance # Necesario para la función de respuesta

    with gr.Blocks(theme=gr.themes.Soft()) as interfaz:
        gr.Markdown("# Chatbot Personalizado con PEFT y Gradio")
        gr.Markdown("Carga un modelo base o un modelo afinado con PEFT, y luego chatea con él.")

        with gr.Row():
            with gr.Column(scale=1):
                modelo_id_input = gr.Textbox(label="ID/Ruta Modelo Base o Ruta Adaptador PEFT", value="gpt2-medium") #gpt2-medium para prueba más robusta
                instrucciones_input = gr.Textbox(label="Instrucciones del Sistema", value="Eres un asistente IA muy útil y creativo.")
                es_personalizado_checkbox = gr.Checkbox(label="¿Cargar desde ruta de adaptador PEFT?", value=False)
                # nombre_modelo_base_peft_input = gr.Textbox(label="Nombre/Ruta del Modelo Base (si carga PEFT y no está en config)", placeholder="Ej: gpt2")
                cargar_button = gr.Button("🚀 Cargar y Configurar Chatbot")
                status_output = gr.Label(label="Estado del Chatbot")

        # La función de carga ahora toma todos los parámetros necesarios
        cargar_button.click(
            fn=lambda id_path, instr, es_pers: inicializar_chatbot_global(id_path, instr, es_pers),
            inputs=[modelo_id_input, instrucciones_input, es_personalizado_checkbox], # , nombre_modelo_base_peft_input
            outputs=status_output
        )
        
        gr.Markdown("---")
        gr.Markdown("## Conversación")
        # Usar gr.ChatInterface dentro de Blocks para más control
        # Necesitamos pasar una función que solo toma (mensaje, historial)
        def chat_fn_wrapper(message, history):

            # Simplificado: chatbot_instance mantiene su propio historial. Gradio solo muestra.
            if chatbot_instance is None:
                return "Error: El chatbot no está inicializado. Por favor, carga un modelo primero."
            
            print(f"Historial Gradio (antes): {history}")
            print(f"Mensaje Usuario (Gradio): {message}")
            respuesta = chatbot_instance.responder(message) # El chatbot interno usa su propio contexto
            print(f"Respuesta Chatbot (Gradio): {respuesta}")
            return respuesta

        gr.ChatInterface(
            fn=chat_fn_wrapper, 
            examples=[["¿Cómo estás hoy?"], ["Explícame qué es un transformador en IA."]],
        )
        
    return interfaz


# --- Flujo Principal ---

def main_entrenamiento_ejemplo(nombre_modelo_base="gpt2", ruta_guardado_peft="./peft_adaptadores"):
    """
    Ejemplo de cómo configurar PEFT y simular un paso de "entrenamiento".
    En un caso real, aquí iría el bucle de entrenamiento con tus datos.
    """
    print("--- Iniciando Flujo de Entrenamiento de Ejemplo con PEFT ---")
    modelo_base, tokenizador_base = cargar_modelo_base(nombre_modelo_base, optimizaciones={"cuantizacion": True, "bits": 4} if dispositivo_global.type == 'cuda' else None)
    
    if modelo_base is None or tokenizador_base is None:
        print("No se pudo cargar el modelo base. Abortando entrenamiento de ejemplo.")
        return

    # Especificar target_modules para gpt2
    # Para gpt2, 'c_attn' es la capa principal de atención donde se aplican Q, K, V.
    # Si se usa otro modelo, estos módulos deben cambiar.
    target_modules_gpt2 = ["c_attn"]

    print(f"Configurando PEFT para el modelo base: {nombre_modelo_base}")
    modelo_peft = configurar_peft(
        modelo_base,
        r=16,  # Rango LoRA, mayor puede dar más capacidad pero más parámetros entrenables
        lora_alpha=32, # Usualmente 2*r
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules_gpt2 if "gpt2" in nombre_modelo_base else None # Ajustar si es otro modelo
    )

    if modelo_peft == modelo_base: # Si configurar_peft falló y devolvió el original
        print("La configuración de PEFT falló. Abortando.")
        return

    print("Simulación de entrenamiento completada (paso omitido).")


    # Guardar los adaptadores PEFT
    guardar_modelo_peft(modelo_peft, tokenizador_base, ruta_guardado_peft, nombre_adaptador=f"{nombre_modelo_base.replace('/','_')}_lora")
    print("--- Fin del Flujo de Entrenamiento de Ejemplo con PEFT ---")

    # Limpiar memoria
    del modelo_base, modelo_peft, tokenizador_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def main_despliegue(modelo_a_cargar="gpt2-medium", # Puede ser un ID de Hugging Face o una ruta a adaptadores PEFT
                    instrucciones="Eres un asistente virtual que ayuda a los usuarios con sus preguntas.",
                    es_modelo_peft=False, # True si modelo_a_cargar es una ruta a adaptadores PEFT
                    nombre_modelo_base_para_peft="gpt2-medium"): # Necesario si es_modelo_peft es True y no está en config
    """
    Función principal para configurar y lanzar la interfaz web del chatbot.
    """
    global chatbot_instance, modelo_global, tokenizador_global # Para que Gradio pueda accederlos
    print("--- Iniciando Despliegue del Chatbot ---")

    # Inicializar el dispositivo global
    verificar_dispositivo()

    # La inicialización del chatbot ahora se hace a través de la UI de Gradio
    # o se podría hacer una carga inicial aquí si se desea.
    
    # Si queremos cargar un modelo por defecto al inicio:
    print(f"Cargando modelo inicial por defecto: {modelo_a_cargar}")

    # Crear y lanzar la interfaz web
    interfaz = crear_interfaz_web()
    print("Lanzando interfaz de Gradio...")
    interfaz.launch()

    print("--- Interfaz de Gradio cerrada ---")


if __name__ == "__main__":
    main_despliegue()
```
![image](https://github.com/user-attachments/assets/367eda5a-a7c6-455d-ba84-863e22b24561)
![image](https://github.com/user-attachments/assets/f8b4ae19-5778-409a-b593-7b0f4e821466)
![image](https://github.com/user-attachments/assets/c4f9266f-4808-4696-bee8-cd64eda81744)

### Preguntas Teóricas
1. **¿Cuáles son las diferencias fundamentales entre los modelos encoder-only, decoder-only y encoder-decoder en el contexto de los chatbots conversacionales? Explique qué tipo de modelo sería más adecuado para cada caso de uso y por qué.**
Los modelos de lenguaje usados en chatbots conversacionales, se dividen principalmente en tres tipos: encoder-only, decoder-only y encoder-decoder. Cada uno tiene características específicas que los hacen más o menos adecuados según el caso de uso.

Modelos Encoder-Only: Estos modelos, como BERT, procesan el texto de entrada en su totalidad de forma bidireccional, es decir, analizan el contexto de cada palabra considerando tanto lo que está antes como después. El caso de uso ideal son como chatbots de soporte técnico o FAQ.

Modelos Decoder-Only: Modelos como GPT o LLaMA procesan el texto de manera unidireccional (de izquierda a derecha) y están optimizados para generar texto de forma secuencial. Cada palabra generada se basa en las anteriores, lo que los hace ideales para producir respuestas coherentes y fluidas. El caso de uso ideal para este modelo funcionan muy bien para chatbots conversacionales como ChatGPT, donde el usuario espera respuestas fluidas, creativas y adaptadas a una conversación abierta.

Modelos Encoder-Decoder: Estos modelos, como T5 o BART, combinan un encoder (para comprender el texto de entrada) y un decoder (para generar texto de salida). El encoder procesa el input y el decoder genera una respuesta basada en esa comprensión. Son muy usados en tareas de transformación de texto, como traducción o resumen. El caso de uso ideal para este modelo son chatbots de traducción, resumen o para tareas estructuradas.

2. **Explique el concepto de "temperatura" en la generación de texto con LLMs. ¿Cómo afecta al comportamiento del chatbot y qué consideraciones debemos tener al ajustar este parámetro para diferentes aplicaciones?**
La temperatura es un parámetro en los modelos de lenguaje grandes (LLMs) que regula cuán aleatorias o creativas serán las respuestas. Con una temperatura baja (cerca de 0), el modelo elige palabras más probables, resultando en texto coherente y factual, perfecto para chatbots de soporte técnico. Con una temperatura alta (cerca de 1 o más), el modelo puede generar respuestas más diversas y sorprendentes, ideal para chats creativos, aunque a veces menos relevantes.
Tareas precisas: Usa temperaturas bajas para documentación legal o médica, asegurando respuestas confiables 
Tareas creativas: Opta por temperaturas altas para poesía o brainstorming

3. **Describa las técnicas principales para reducir el problema de "alucinaciones" en chatbots basados en LLMs. ¿Qué estrategias podemos implementar a nivel de inferencia y a nivel de prompt engineering para mejorar la precisión factual de las respuestas?**
Las alucinaciones en los chatbots basados en modelos de lenguaje grande (LLMs) se refieren a respuestas incorrectas, inventadas o engañosas que aparentan ser coherentes y verídicas. Reducir este fenómeno requiere una combinación de técnicas a diferentes niveles del diseño y uso del modelo.
A nivel de inferencia:
1. RAG (Retrieval-Augmented Generation):
Esta técnica combina el poder de generación del modelo con una base externa de conocimientos. Antes de generar una respuesta, el sistema busca información en una base de datos, documentos o sitios web confiables, y luego utiliza esos fragmentos para fundamentar su respuesta. Esto limita las invenciones del modelo al anclarlo en hechos verificados.

2. Verificación cruzada (multi-hop verification):
Consiste en generar múltiples respuestas sobre la misma consulta desde diferentes perspectivas o modelos y luego analizarlas para extraer puntos en común. Si varias variantes coinciden en un hecho, es más probable que sea correcto.

3. Filtros post-inferencia:
Se pueden aplicar validaciones automáticas que analicen si las afirmaciones hechas por el modelo coinciden con una base de conocimiento estructurada (como una base de datos o API). En caso contrario, se descartan o ajustan.

A nivel de prompt engineering:
1. Instrucciones claras y específicas:
Mientras más preciso sea el prompt, menor es la ambigüedad en la respuesta. 

2. Uso de avisos metacognitivos:
Se le puede pedir explícitamente al modelo que razone paso a paso, verifique su respuesta antes de darla o que indique cuando no sabe algo.

3. Proporcionar contexto estructurado:
Si el modelo recibe una base de conocimiento o un conjunto de hechos relevantes antes de la pregunta, es más probable que los utilice como referencia y no invente.
