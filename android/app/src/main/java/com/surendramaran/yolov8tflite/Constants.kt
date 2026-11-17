package com.surendramaran.yolov8tflite

object Constants {
    const val MODEL_PATH = "best.tflite"
    val LABELS_PATH: String? = "labels.txt" // provide your labels.txt file if the metadata not present in the model
}
