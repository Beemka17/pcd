import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache_resource()
def load_model():
	model = tf.keras.models.load_model('./klasifikasi_ban.hdf5',compile=False)
	return model


def predict_class(image, model):

	image = tf.cast(image, tf.float32)
	image = tf.image.resize(image, [180, 180])

	image = np.expand_dims(image, axis = 0)

	prediction = model.predict(image)

	return prediction


model = load_model()
st.title('Prediksi Kualitas Ban')

file = st.file_uploader("Upload gambar Ban (Hanya Foto Ban Saja)", type=["jpg", "png","jpeg"])


if file is None:
	st.text('Silahkan masukkan gambar ban...')

else:
	slot = st.empty()
	slot.text('Tunggu sebentar....')

	test_image = Image.open(file)

	st.image(test_image, caption="Input Image", width = 400)

	pred = predict_class(np.asarray(test_image), model)

	class_names = ['defective', 'good']

	class_desc = {
		'defective': 'Ban ini sudah tidak layak pakai',
		'good' : 'Ban ini masih llayak pakai',
	}

	result = class_names[np.argmax(pred)]

	output = 'Kualitas ban ini adalah: ' + result

	slot.text('Done')

	st.success(output)