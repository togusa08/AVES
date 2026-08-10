import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI()

class_names = ["anna_hummingbird", "arctic_tern", "blue_jay",
                "cardinal","downy_woodpecker","goldfinch",
                "herring_gull","hooded_oriole" ,"house_sparrow" ,
                "kingfisher" ,"large_billed_crow", "mockingbird",
                "purple_finch", "rock_dove" ,"yellow_headed_blackbird"] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load("bird_classifier.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# HTML/CSS/JS CODE

@app.get("/", response_class=HTMLResponse)
def main_page():
    html_content = """
    <html>
<head>
    <title>AVES</title>
    <link href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;700&display=swap" rel="stylesheet">
    <style>
        /* ====== COLOUR PALETTE ====== */
        :root {
            --dark-green: #004942;
            --green: #00695c;
            --mint: #12bfa0;
            --aqua: #69d1cc;
            --light: #baf5f3;
        }

        /* ====== BASE ====== */
        body {
            font-family: 'Spectral', serif;
            text-align: center;
            margin: 0;
            background-color: var(--dark-green);
            color: white;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        /* ====== NAV BAR ====== */
        nav {
            background-color: var(--green);
            padding: 16px 0;
            display: flex;
            justify-content: center;
            gap: 80px;
            font-size: 19px;
            letter-spacing: 1px;
            width: 100%;
        }

        nav a {
            color: white;
            text-decoration: none;
            font-weight: 600;
        }

        nav a:hover {
            text-decoration: underline;
        }

        /* ====== MAIN SECTION ====== */
        h1 {
            margin-top: 30px;
            font-size: 30px;
            font-weight: 200;
            letter-spacing: 3px;
        }

        .container {
            display: flex;
            gap: 40px;
            margin-top: 60px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .dropzone, .preview-box, .info-box {
            background-color: var(--mint);
            width: 280px;
            min-height: 280px;
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            cursor: pointer;
            transition: all 0.3s ease, opacity 0.5s ease;
            opacity: 1;
        }

        .dropzone:hover {
            background-color: var(--aqua);
        }

        .arrow-circle {
            background-color: var(--aqua);
            width: 120px;
            height: 120px;
            border-radius: 36%;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .arrow {
            font-size: 50px;
            color: var(--dark-green);
            font-weight: bold;
            transform: translateY(-3px);
        }

        .dragtext {
            font-size: 18px;
            margin-top: 14px;
            letter-spacing: 1.5px;
        }

        #d {
            margin-top: 20px;
            font-size: 20px;
            color: var(--light);
        }

        #result {
            margin-top: 20px;
            font-size: 16px;
            color: var(--light);
        }

        .preview-box img {
            max-width: 90%;
            max-height: 90%;
            border-radius: 16px;
        }

        .info-box {
            background-color: var(--aqua);
            opacity: 0;
            padding: 16px;
            color: var(--dark-green);
            text-align: left;
        }

    </style>
</head>

<body>
    <nav>
        <a href="#">Avian Visual Evaluation System</a>
    </nav>

    <h1>AVES</h1>

    <div class="container">
        <div class="dropzone" id="dropzone">
            <div class="arrow-circle">
                <div class="arrow">↑</div>
            </div>
            <p id="d"> DRAG & DROP or CLICK </p>
        </div>

        <div class="preview-box" id="preview-box" style="opacity:0;">
            #shows image uploaded
        </div>

        <div class="info-box" id="info-box">
            #shows info about bird
        </div>
    </div>

    <p id="result"></p>


    <script>
        const dropzone = document.getElementById("dropzone");
        const result = document.getElementById("result");
        const previewBox = document.getElementById("preview-box");
        const infoBox = document.getElementById("info-box");

        // ===================== BIRD INFO =====================
        const birdInfo = {
            "anna_hummingbird": {
                temperature: "20-30°C",
                vegetation: "Gardens, shrubs, forests",
                funFact: "They can hover like helicopters!"
            },
            "arctic_tern": {
                temperature: "-10 to 5°C",
                vegetation: "Coastal areas, tundra",
                funFact: "Migrates the longest distance of any bird."
            },
            "blue_jay": {
                temperature: "10-25°C",
                vegetation: "Forests, urban areas",
                funFact: "Known for their intelligence and complex social behavior."
            },
            "cardinal": {
                temperature: "15-30°C",
                vegetation: "Woodlands, gardens",
                funFact: "Males are bright red, females are duller."
            },
            "downy_woodpecker": {
                temperature: "5-25°C",
                vegetation: "Forests, wooded areas",
                funFact: "They drum on trees to communicate."
            },
            "goldfinch": {
                temperature: "15-30°C",
                vegetation: "Fields, gardens",
                funFact: "They eat mainly seeds."
            },
            "herring_gull": {
                temperature: "5-20°C",
                vegetation: "Coastal areas",
                funFact: "They are opportunistic feeders."
            },
            "hooded_oriole": {
                temperature: "15-30°C",
                vegetation: "Open woodlands",
                funFact: "Males have bright orange plumage."
            },
            "house_sparrow": {
                temperature: "10-30°C",
                vegetation: "Urban areas",
                funFact: "Adapted to living close to humans."
            },
            "kingfisher": {
                temperature: "20-30°C",
                vegetation: "Rivers, lakes, wetlands",
                funFact: "Excellent fish hunters."
            },
            "large_billed_crow": {
                temperature: "10-25°C",
                vegetation: "Forests, urban areas",
                funFact: "Very intelligent and uses tools."
            },
            "mockingbird": {
                temperature: "15-30°C",
                vegetation: "Open woodlands, gardens",
                funFact: "Can mimic the sounds of other birds."
            },
            "purple_finch": {
                temperature: "10-25°C",
                vegetation: "Forests",
                funFact: "Males have a raspberry-red color."
            },
            "rock_dove": {
                temperature: "10-30°C",
                vegetation: "Urban areas, cliffs",
                funFact: "Common city pigeon."
            },
            "yellow_headed_blackbird": {
                temperature: "10-25°C",
                vegetation: "Marshes, wetlands",
                funFact: "Males have striking yellow heads."
            }
        };

        // ===================== HELPERS =====================
        function showPreview(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewBox.innerHTML = "<img src='" + e.target.result + "' />";
                previewBox.style.opacity = "1";
            };
            reader.readAsDataURL(file);
        }

        function showInfo(className) {
            if (birdInfo[className]) {
                const info = birdInfo[className];
                infoBox.innerHTML = `
                    <b>Temperature:</b> ${info.temperature}<br>
                    <b>Vegetation:</b> ${info.vegetation}<br>
                    <b>Fun Fact:</b> ${info.funFact}
                `;
                infoBox.style.opacity = "1";
            } else {
                infoBox.innerHTML = "No info available";
                infoBox.style.opacity = "1";
            }
        }

        function handleFile(file) {
            showPreview(file);
            const formData = new FormData();
            formData.append("file", file);

            fetch("/predict/", { method: "POST", body: formData })
                .then(res => res.json())
                .then(data => {
                    result.innerHTML = "Prediction: <b>" + data.class + "</b> (confidence: " + data.confidence.toFixed(2) + ")";
                    showInfo(data.class);
                });
        }

        // ===================== EVENTS =====================
        dropzone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropzone.style.background = "#69d1cc";
        });

        dropzone.addEventListener("dragleave", () => {
            dropzone.style.background = "#12bfa0";
        });

        dropzone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropzone.style.background = "#12bfa0";
            const file = e.dataTransfer.files[0];
            handleFile(file);
        });

        dropzone.addEventListener("click", () => {
            const input = document.createElement("input");
            input.type = "file";
            input.accept = "image/*";
            input.onchange = (e) => {
                const file = e.target.files[0];
                handleFile(file);
            };
            input.click();
        });

    </script>
</body>
</html> """

    return HTMLResponse(content=html_content)


@app.post("/predict/")
async def predict(file: UploadFile):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_t = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, pred = torch.max(probs, dim=0)

    return JSONResponse({"class": class_names[pred.item()], "confidence" : conf.item()*100})