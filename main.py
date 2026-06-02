
from fastapi import FastAPI
from fastapi import  status, Body
from fastapi.responses import JSONResponse

from services.sentiment_analysis import sentiment_service
from services.preprocessing import preprocess_text

app= FastAPI()

from pyngrok import ngrok

# 1. قفل أي جلسة قديمة معلقة في الخلفية
# ngrok.kill()

# # 2. فتح الاتصال بالدومين الثابت بتاعك
# public_url = ngrok.connect(8000, domain="unhumble-downrightly-santa.ngrok-free.dev")

# print("\n🚀 ngrok Tunnel Started Successfully!")
# print("🔗 Public URL:", public_url)

@app.get("/")
def welcom():
    return{"message": "Welcom To Text Sentiment Analysis"}

@app.post("/sentiment")
async def analyze_text(text: str = Body(..., embed=True)):
    try :
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
                "signal": "TEXT_FAILD"
            }
        )
    

if __name__ == "__main__":
    import uvicorn
    import os
    # Railway بتمرر البورت في متغير بيئي اسمه PORT
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)