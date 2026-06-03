import os
import gdown
from fastapi import FastAPI, status, Body
from fastapi.responses import JSONResponse

# -----------------------------------------------------------------
# 📥 كود تنزيل الأوزان تلقائياً من جوجل درايف داخل فولدر best2
# -----------------------------------------------------------------
weights_dir = 'best2'
weights_path = os.path.join(weights_dir, 'model.safetensors') 

# التأكد من وجود الفولدر أولاً
os.makedirs(weights_dir, exist_ok=True)

if not os.path.exists(weights_path):
    print("📥 Railway detected missing weights. Downloading model.safetensors from Google Drive...")
    # الـ ID بتاع ملف الـ safetensors الصافي اللي فكينا ضغطه
    file_id = '11eGrKH_CflbhvD1Xi7oLuxqrEgtM3hZi' 
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # تنزيل الملف مباشرة داخل الفولدر
    gdown.download(url, weights_path, quiet=False)
    print("✅ Download completed successfully!")
else:
    print("✅ Weights already exist in best2 folder. Booting up...")
# -----------------------------------------------------------------

# استيراد الخدمات بعد التأكد من تحميل الأوزان
from services.sentiment_analysis import sentiment_service
from services.preprocessing import preprocess_text

app = FastAPI()

@app.get("/")
def welcom():
    return {"message": "Welcome To Text Sentiment Analysis"}

@app.post("/sentiment")
async def analyze_text(text: str = Body(..., embed=True)):
    try:
        clean_text = preprocess_text(text)
        sentiment_values = sentiment_service.analyze_mentee_review(clean_text)

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
               "TEXT": text,
               "Cleand_Text": clean_text,
               "Sentiment_Label": sentiment_values,
            }
        )

    except Exception as e:
        print(e)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": "TEXT_FAILED"
            }
        )

if __name__ == "__main__":
    import uvicorn
    # Railway بتمرر البورت في متغير بيئي اسمه PORT
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
