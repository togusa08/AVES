<p align="center">
  <img src="https://github.com/togusa08/AVES/blob/main/image/aves_logo_transparent.png" width="250px">
</p>
<h2 align="center">Fast and lightweight image classification tool for bird</h2>
Avian Visual Evaluation System is an image classifier and web application system made for identifying birds species from uploaded images using MobileNetV2 deep learning module trained with PyTorch.

## Feature
+  Fast and light weight prediction for major species of bird
+  Drag-and-drop image upload
+  Prediction confidence score
+ Fun facts of the predicted species
+ FastAPI backend

## Demo
Deplyment: [aves-bird-classifier.onrender.com](https://aves-bird-classifier.onrender.com)

Checkout the video on: [AVES Demo Video](https://youtu.be/YhA18Dy0TCE)

## Install

### 1.  Clone the repository
``` bash  
git clone https://github.com/togusa08/AVES.git
cd AVES
```
### 2.  Create a virtual environment
#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```
#### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3.  Install dependencies
```
pip install -r requirements.txt
```
### 4.  Start the FastAPI server
```
uvicorn web:app --reload
```
The application should then be available at:
```http://127.0.0.1:8000```

## Reference
`https://github.com/banesullivan/README`

`https://qiita.com/omiita/items/77dadd5a7b16a104df83`
