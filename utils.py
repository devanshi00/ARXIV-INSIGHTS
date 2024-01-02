import random
import arxiv
import datetime
import os
import requests
import subprocess
import json
import re

# Set your search parameters
search_queries = ['cat:cs.CL','cat:cs.DC','cat:cs.IR','cat:cs.CY','cat:cs.CV','cat:cs.AI','cat:cs.GR']  # Modify or add more subcategories as per your requirement


max_results_per_query = 3 # Maximum number of papers to download for each subcategory
num_papers_to_download=1

# Get today's date
today = datetime.date.today()
# Get the date a week ago
week_ago = today - datetime.timedelta(weeks=1)

# Create a folder for the papers
folder_name = 'arxiv'
os.makedirs(folder_name, exist_ok=True)

# Function to download the PDF from a given URL
def download_pdf(url, file_path):
    response = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(response.content)

# Collect all the paper URLs from the search queries
all_pdf_urls = []
for search_query in search_queries:
    query = f"submittedDate:[{week_ago.year:04d}-{week_ago.month:02d}-{week_ago.day:02d} TO {today.year:04d}-{today.month:02d}-{today.day:02d}] {search_query}"  # Add lang:en to filter English papers
    results = arxiv.Search(
        query=query,
        max_results=max_results_per_query,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    ).results()

    pdf_urls = [(result.pdf_url, result.title) for result in results]
    all_pdf_urls.extend(pdf_urls)

if num_papers_to_download < 0:
    num_papers_to_download = 0  # or set it to some default value
selected_pdf_urls = random.sample(all_pdf_urls, num_papers_to_download)

# Download the selected papers
for i, (pdf_url, title) in enumerate(selected_pdf_urls):
    file_path = os.path.join(folder_name, f'{title}.pdf')
    download_pdf(pdf_url, file_path)

print(f"Number of papers downloaded: {num_papers_to_download}")
print(f"Papers are saved in the '{folder_name}' folder.")



# output_folder = './json'
# folder = './arxiv'

# # Create the output folder if it doesn't exist
# os.makedirs(output_folder, exist_ok=True)

# # Run Science Parse CLI for each PDF to generate JSON output
# for i, pdf_file in enumerate(os.listdir(folder)):
#     if pdf_file.endswith(".pdf"):
#         cmd = ["java", "-Xmx6g", "-jar", "science-parse-cli-assembly-2.0.3.jar", "./arxiv", "-o", "./json"]
#         subprocess.run(cmd, check=True)
# for i, pdf_file in enumerate(os.listdir(output_folder)):
#   if pdf_file.endswith(".pdf.json"):
#     json_file_name = pdf_file[:-8] + "json"
#     json_file_path = os.path.join(output_folder, json_file_name)
#     os.rename(os.path.join(output_folder, pdf_file), json_file_path)
    
# import os
# from pathlib import Path

# from transformers import AutoImageProcessor, TableTransformerForObjectDetection
# import torch
# from PIL import Image
# import fitz  # Import pymupdf

# # Specify the local folder path containing PDFs
# pdf_folder = "./arxiv"

# # Create a folder to store the extracted tables
# output_folder = "./extracted_tables"
# os.makedirs(output_folder, exist_ok=True)

# # Load the pre-trained models
# image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
# model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

# # Iterate over each PDF file in the folder
# for pdf_file in Path(pdf_folder).glob("*.pdf"):
#     pdf_path = str(pdf_file)

#     # Load the PDF document
#     doc = fitz.open(pdf_path)

#     # Create a folder for the current PDF
#     pdf_output_folder = os.path.join(output_folder, pdf_file.stem)
#     os.makedirs(pdf_output_folder, exist_ok=True)

#     # Process each page of the PDF
#     for page_num in range(doc.page_count):
#         page = doc.load_page(page_num)  # Load each page of the PDF
#         pix = page.get_pixmap()
#         image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

#         inputs = image_processor(images=image, return_tensors="pt")
#         outputs = model(**inputs)

#         # Convert outputs (bounding boxes and class logits) to COCO API
#         target_sizes = torch.tensor([image.size[::-1]])
#         results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

#         # Iterate over each detected table and save them as PNG
#         for i, bbox in enumerate(results["boxes"]):
#             table_bbox = bbox.tolist()

#             # Add buffer to the table bounding box
#             x_min, y_min, x_max, y_max = table_bbox
#             x_min -= 15
#             y_min -= 15
#             x_max += 15
#             y_max += 15

#             # Adjust the coordinates to stay within image boundaries
#             x_min = max(0, x_min)
#             y_min = max(0, y_min)
#             x_max = min(image.width, x_max)
#             y_max = min(image.height, y_max)

#             table_image = image.crop((x_min, y_min, x_max, y_max))

#             # Save the table image as PNG in the current PDF's output folder
#             table_image.save(os.path.join(pdf_output_folder, f"detected_table_{page_num + 1}_{i + 1}.png"))

#     # Close the PDF document
#     doc.close()

# import os
# import json
# import re
# import torch
# from transformers import pipeline

# # Get the list of JSON files in the folder
# json_folder = "./json"
# json_files = [os.path.join(json_folder, filename) for filename in os.listdir(json_folder) if filename.endswith('.json')]
# # Initialize the summarization pipeline with the BART model
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# # Convert the model to CUDA if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# summarizer.model.to(device)

# # Create a folder to store the summary JSON files
# summary_folder = "./summary_json"
# os.makedirs(summary_folder, exist_ok=True)

# # Iterate over the JSON files and extract sections, then generate summaries
# for json_file in json_files:
#     # Read the JSON file
#     with open(json_file, 'r') as f:
#         json_data = json.load(f)

#     # Extract sections and their text
#     sections = json_data['metadata']['sections']
#     extracted_sections = []

#     # Iterate over the sections and store the section name and text in a list
#     for section in sections:
#         section_name = section['heading']
#         section_text = section['text']
#         if section_name is None:
#             continue  # Skip this section if the section_name is None
#         extracted_sections.append({"heading": section_name, "text": section_text})

#     # Regular expression pattern to match arrays of various sizes containing only numbers
#     pattern = r'\[(?:\d+(?:,\s*)?)+\]'

#     print("Summaries for", json_file)
#     # Print the extracted sections without the arrays of numbers and generate summaries
#     section_summaries = []
#     for section in extracted_sections:
#         section_name = section["heading"]
#         section_text = section["text"]
#         print("Section:", section_name)
#         if section_text:
#             # Remove the arrays of numbers from the section text
#             modified_text = re.sub(pattern, '', section_text)

#             # Split the text into smaller chunks for summarization
#             max_input_length = summarizer.tokenizer.model_max_length
#             chunked_text = [modified_text[i:i + max_input_length] for i in range(0, len(modified_text), max_input_length)]

#             summaries = []
#             for chunk in chunked_text:
#                 # Adjust input length based on the number of tokens in the chunk
#                 input_length = len(summarizer.tokenizer.tokenize(chunk))
#                 min_length = 0
#                 max_length = min(input_length * 2, summarizer.model.config.max_length)

#                 # Move input tensors to CUDA if available
#                 inputs = summarizer.tokenizer(chunk, truncation=True, return_tensors="pt").to(device)

#                 summary = summarizer.model.generate(
#                     inputs.input_ids.to(device),
#                     attention_mask=inputs.attention_mask.to(device),
#                     min_length=min_length,
#                     max_length=max_length
#                 )

#                 # Move the generated summary to CPU for further processing
#                 summary = summarizer.tokenizer.decode(summary[0].to("cpu"), skip_special_tokens=True)
#                 summaries.append(summary)

#             # Join the summaries of the chunks into a single summary
#             final_summary = ' '.join(summaries)

#             section_summary = {
#                 "heading": section_name,
#                 "text": final_summary
#             }
#             section_summaries.append(section_summary)

#             print("Summary:", final_summary)

#     # Create a dictionary for the summary JSON file
#     summary_dict = {
#         "name": os.path.splitext(os.path.basename(json_file))[0] + "_summary.json",
#         "metadata": {
#             "title": json_data['metadata']['title'],
#             "sections": section_summaries
#         }
#     }

#     # Generate the file name for the summary JSON file
#     summary_file_name = os.path.splitext(os.path.basename(json_file))[0] + "_summary.json"
#     summary_file_path = os.path.join(summary_folder, summary_file_name)

#     # Write the summary JSON file
#     with open(summary_file_path, 'w') as summary_file:
#         json.dump(summary_dict, summary_file, indent=4)

#     print()

# print("Extraction and summarization completed!")