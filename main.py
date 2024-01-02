
import json
import os
from PIL import Image
import streamlit as st
import os
import requests
import tempfile
import torch
import re
import subprocess
from transformers import pipeline
import base64
import datetime


# Get the list of JSON files in the "summary" folder
summary_folder = "./summary_json"
json_files = [file for file in os.listdir(summary_folder) if file.endswith(".json")]
# Get the list of extracted tables folders
tables_folder = "./extracted_tables"
tables_folders = [folder for folder in os.listdir(tables_folder) if os.path.isdir(os.path.join(tables_folder, folder))]

# Set the initial scroll position at the top
st.markdown(
    """
    <style>
    body {
        margin: 0;
        padding: 0;
        overflow-y: scroll;
    }
    </style>
    """,
    unsafe_allow_html=True)

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from the root folder and set as bg.

    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"


    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover;
             color: white; /* Change the font color to black */
         }}
         .title-container {{
             padding-top: 0px;
             padding-bottom: 0px;
         }}
         .subtitle-container {{
             padding-bottom: 10px;
         }}
          .title, p, h1, h2, h3, h4, h5, h6 {{
           color: black; /* Change the font color to black */
         }}


         .choose-title, .css-2trqyj {{
             color: black;
         }}
         .subtitle {{
             font-size: 50px;
             font-family: 'Brush Script MT', cursive;
             color: black;
         }}
         </style>
         """,
         unsafe_allow_html=True
    )

# Set the PNG image as the background of the page
set_bg_hack('/gjhekhebfebfkefej (5).png')



# Set up the page navigation

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"

# Function to navigate to the ARXIV SUMMARIZER page
def navigate_to_url_page():
    st.session_state["current_page"] = "url_page"

# Function to navigate to the TITLE PAGE
def navigate_to_title_page():
    st.session_state["current_page"] = "title_page"

# Home page - Display the title buttons for each file

def home():
    # Display the Canva image at the top of the page
    canva_image_path = "/arXiv (1).png"  # Replace with the actual file path of your Canva image
    canva_image = Image.open(canva_image_path)
    st.image(canva_image, use_column_width=True)
    # Center the date and make it bold using CSS
    st.markdown(
        f"""
        <div style="text-align:center; font-weight: bold; font-size: 24px">
        {datetime.datetime.now().strftime('%d %B, %Y')}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Display the date above the "Daily Papers" heading
    today_date = datetime.datetime.now().strftime('%d %B, %Y')
    st.write(f"**{today_date}**")








    st.markdown("<div class='title-container'><h1 class='title'></h1></div>", unsafe_allow_html=True)
    st.markdown("<div class='title-container'><p class='subtitle'>Daily Papers</p></div>", unsafe_allow_html=True)
    # Get the list of JSON files in the "json" folder within the "Files" section of Google Colab
    json_folder = "./json"  # Replace "/json" with the actual path of your "json" folder
    json_files = [file for file in os.listdir(json_folder) if file.endswith(".json")]

    # Iterate over the JSON files and create a button for each file title
    for json_file in json_files:
        with open(os.path.join(json_folder, json_file)) as file:
            data = json.load(file)
            title = data["metadata"]["title"]
            authors = data["metadata"]["authors"] if "authors" in data["metadata"] else []

            # Display the title of the paper
            st.markdown("<div class='title-container'><h2 class='paper-title'>{}</h2></div>".format(title), unsafe_allow_html=True)

            # Display the first four authors' names for the paper
            if len(authors) > 0:
                st.subheader("Authors:")
                first_four_authors = ", ".join(authors[:4])
                st.write(first_four_authors)

            # Create a "Read Summary" button for each file title
            if st.button("Read Summary", key=f"button_{title}"):
                # Set the selected title in the session state
                st.session_state["selected_title"] = title
                # Navigate to the second page
                navigate_to_title_page()


    st.markdown("<div class='title-container'><p class='subtitle'>Wanna Try the Summariser ?</p></div>", unsafe_allow_html=True)
    if st.button("ARXIV SUMMARIZER", key="summarise"):
        navigate_to_url_page()

# Function to download the paper from a given URL and save it in a runtime folder
def download_paper_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Get the filename from the URL or use a default name
        filename = os.path.basename(url).split("?")[0]
        temp_folder = tempfile.mkdtemp()
        file_path = os.path.join(temp_folder, filename)
        with open(file_path, 'wb') as f:
            f.write(response.content)
        return filename, temp_folder
    else:
        raise Exception("Failed to download the paper. Please check the URL and try again.")
# Function to parse the paper using Science Parse and extract sections in JSON format
def parse_paper_with_science_parse(pdf_file_path, output_folder):
    jar_file_path = './science-parse-cli-assembly-2.0.3.jar'
    subprocess.run(['java', '-Xmx6g', '-jar', jar_file_path, pdf_file_path, '-o', output_folder], check=True)

def url_page():
    set_bg_hack('/gjhekhebfebfkefej (5).png')

# Display the "Back to Home" button
    if st.button("Back to Home"):
     st.session_state["current_page"] = "home"
    st.title("Arxiv Paper Summarizer")

    # URL input
    st.write("Enter the Arxiv Paper URL:")
    st.warning("Warning: The URL should be of the form:https://arxiv.org/pdf/{arxiv-id}.pdf")
    url_input = st.text_input("", "Arxiv Paper URL", key="arxiv_url")

    if st.button("Summarize Paper"):
        if url_input:
            try:
                filename, temp_folder = download_paper_from_url(url_input)
                st.success(f"Paper downloaded successfully.Please wait for a few minutes.")

                # Parse the downloaded paper and save the parsed JSON in the same temporary folder
                parse_paper_with_science_parse(temp_folder, temp_folder)

                # Read the parsed JSON file and extract the title and sections
                json_file_name = f"{filename}.json"
                final_json_file_path = os.path.join(temp_folder, json_file_name)
                with open(final_json_file_path, 'r') as f:
                    parsed_data = json.load(f)

                # Display the paper title
                paper_title = parsed_data['metadata']['title']

                st.header(paper_title)

                # Extract sections and their text from the parsed JSON
                sections = parsed_data['metadata']['sections']
                extracted_sections = []
                for section in sections:
                    section_name = section['heading']
                    section_text = section['text']
                    if section_name is not None and section_text:
                        extracted_sections.append({"heading": section_name, "text": section_text})

                # Regular expression pattern to match arrays of various sizes containing only numbers
                pattern = r'\[(?:\d+(?:,\s*)?)+\]'

                # Model for summarization
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                # Convert the model to CUDA if available
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                summarizer.model.to(device)

                # Iterate over the extracted sections, generate summaries, and display the results
                for section in extracted_sections:
                    section_name = section["heading"]
                    section_text = section["text"]
                    st.subheader(section_name)
                    #st.write(section_text)

                    # Remove the arrays of numbers from the section text
                    modified_text = re.sub(pattern, '', section_text)

                    # Generate summaries for smaller chunks of text
                    max_input_length = summarizer.tokenizer.model_max_length
                    chunked_text = [modified_text[i:i + max_input_length] for i in range(0, len(modified_text), max_input_length)]

                    section_summaries = []
                    for chunk in chunked_text:
                        # Adjust input length based on the number of tokens in the chunk
                        input_length = len(summarizer.tokenizer.tokenize(chunk))
                        min_length = 0
                        max_length = min(input_length * 2, summarizer.model.config.max_length)

                        # Move input tensors to CUDA if available
                        inputs = summarizer.tokenizer(chunk, truncation=True, return_tensors="pt").to(device)

                        summary = summarizer.model.generate(
                            inputs.input_ids.to(device),
                            attention_mask=inputs.attention_mask.to(device),
                            min_length=min_length,
                            max_length=max_length
                        )

                        # Move the generated summary to CPU for further processing
                        summary = summarizer.tokenizer.decode(summary[0].to("cpu"), skip_special_tokens=True)
                        section_summaries.append(summary)

                    # Join the summaries of the chunks into a single summary for the section
                    final_summary = ' '.join(section_summaries)

                    # Display the summary for the section
                    st.subheader("Summary:")
                    st.write(final_summary)

            except Exception as e:
                st.error(f"Error: {str(e)}")
# Second page - Display the selected title and its sections
def title_page(title):
    set_bg_hack('/gjhekhebfebfkefej (4).png')
    if st.button("Back to Home"):
        st.session_state.pop("selected_title")

    st.title(title)
    selected_sections = None
    selected_json_data = None

    for json_file in json_files:
        with open(os.path.join(summary_folder, json_file)) as file:
            data = json.load(file)
            if data["metadata"]["title"] == title:
                selected_json_data = data
                selected_sections = data["metadata"]["sections"]
                break

    if selected_sections:
        # Display the image
        # image_path = os.path.join("/content/diagramgpt", f"{title}.png")
        # if os.path.exists(image_path):
        #     image = Image.open(image_path)
        #     st.image(image)

        for section in selected_sections:
            section_title = section["heading"]
            st.subheader(section_title)
            st.write(section["text"])

    # Add a "View PDF" button to download the PDF file
        pdf_path = os.path.join("./arxiv", f"{title}.pdf")
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as pdf_file:
                pdf_content = pdf_file.read()
                st.download_button(label="Download Full Text PDF", data=pdf_content, file_name=f"summary_{title}.pdf")
        else:
            st.write("PDF file not available for this paper.")
    # Button to generate and display the tables
    if st.button("Show Tables"):
        display_tables(title)


import re
def display_tables(title):
    # Find the subfolder corresponding to the selected title
    subfolder = None
    for folder in tables_folders:
        if folder == title:
            subfolder = os.path.join(tables_folder, folder)
            break

    if subfolder:
        table_files = os.listdir(subfolder)
        table_files_sorted = sorted(table_files, key=lambda x: int(re.search(r'\d+', x).group()))

        if table_files_sorted:
            st.subheader("Extracted Tables")

            # Display the tables in pairs
            table_counter = 1
            for i in range(0, len(table_files_sorted), 2):
                col1, col2 = st.columns(2)

                # Display table 1
                table_path1 = os.path.join(subfolder, table_files_sorted[i])
                table_image1 = Image.open(table_path1)
                col1.image(table_image1, use_column_width=True, caption=f"Table {table_counter}")
                table_counter += 1

                # Display table 2 if exists
                if i + 1 < len(table_files_sorted):
                    table_path2 = os.path.join(subfolder, table_files_sorted[i + 1])
                    table_image2 = Image.open(table_path2)
                    col2.image(table_image2, use_column_width=True, caption=f"Table {table_counter}")
                    table_counter += 1

    else:
        st.info("No tables extracted for this PDF.")

# Retrieve the selected title from the session state
selected_title = st.session_state.get("selected_title")
selected_url=st.session_state.get("url_page")

# Sidebar - Display information, links, and navigation options
st.sidebar.title("Literature Review Generator 📑")
st.sidebar.caption("Get Latest Scientific Papers summarised on a click")

st.sidebar.caption("Look behind the scenes of Literature Review Generator [here](https://colab.research.google.com/drive/1ANozsys-P6ckj_gABFZO9FPMinGdmtLr#scrollTo=zwC0r_VlFtmU).")
st.sidebar.markdown("---")
# Check the current page and display the corresponding content
if st.session_state["current_page"] == "url_page":
    url_page()
elif st.session_state["current_page"] == "title_page":
    selected_title = st.session_state.get("selected_title")
    if selected_title:
        title_page(selected_title)
    else:
        st.session_state["current_page"] = "home"
        home()
else:
    home()
