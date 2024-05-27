from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
from PIL import Image
import uvicorn
from io import BytesIO
from bgRomov import remove_background

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory=".")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/remove_background")
async def remove_background_endpoint(file: UploadFile = File(...)):
    file_location = f"static/{file.filename}"
    print(f"Received file: {file.filename}")
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
        print(f"File saved to {file_location}")

    output_path = f"static/output_{file.filename.split('.')[0]}.png" 
    print(f"Calling remove_background with input: {file_location}, output: {output_path}")
    remove_background(file_location, output_path)

    # # yesma RGBA image to RGB convert garey ko
    # im = Image.open(output_path)
    # rgb_im = im.convert('RGB')
    # rgb_im.save(output_path)

    # return FileResponse(output_path)

    with open(output_path, "rb") as output_file:
        processed_image_data = output_file.read()

    return StreamingResponse(BytesIO(processed_image_data), media_type="image/png", headers={"Content-Disposition": f"attachment; filename={file.filename}"})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)