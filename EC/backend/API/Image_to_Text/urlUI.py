### Flask Streamlit UI with both image and URL ###

import streamlit as st
import io
import torch
import requests
from flask import request
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

def process_image_to_text(image):
    model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224")

    # Add these lines to force CPU usage
    device = torch.device("cpu")
    model = model.to(device)

    processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224" , force_reload=True)

    prompt = "<grounding>An image of"

    # Save the image with the correct extension
    image.save("new_image.png")

    # Reload the image with the correct extension
    image = Image.open("new_image.png")

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=256,
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # By default, the generated text is cleaned up, and the entities are extracted.
    processed_text = processor.post_process_generation(generated_text)[0]

    return processed_text

def main():
    st.markdown("<h1 style='text-align: center; color: purple;'>Image to Text</h1>",unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Application to convert an Image to Text</h4>",unsafe_allow_html=True)

    # Help Section
    st.sidebar.subheader("Help")
    with st.sidebar.expander("How to Use"):
         st.write("   - **Upload an image file:** Use the 'Browse files' button to upload an image file.")
         st.write("   - **Supported formats:'jpeg , png'**" )
         st.write("   - **Image size**: Restricted to 200Mb ")

    st.header("Choose an image or enter a URL...")
    option = st.radio("Select input type:", ("Image", "URL"))


    if option == "Image":
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpeg"])
        # Add a button to trigger text generation
        if st.button("Generate Text"):
            if uploaded_file is not None:
                # Process the uploaded image
                image = Image.open(io.BytesIO(uploaded_file.read()))
                st.image(image, caption="Uploaded Image.", use_column_width=True)

                # Perform image-to-text processing
                processed_text = process_image_to_text(image)

                st.header("Processed Text:")
                st.write(processed_text)
                

    elif option == "URL":
        image_url = st.text_input("Enter Image URL:")
        if st.button("Generate Text"):
            try:
                # Download image from URL
                image = Image.open(io.BytesIO(requests.get(image_url).content))

                # Perform image-to-text processing
                processed_text = process_image_to_text(image)

                st.header("Processed Text:")
                st.write(processed_text)

            except Exception as e:
                st.error("Error processing image:", e)

if __name__ =="__main__":
    main ()
