#!/usr/bin/env python
import requests
import time
import os
from bs4 import BeautifulSoup

# Number of catalog pages to crawl. Each page on Gutenberg typically lists ~25 books.
# WARNING: Increasing NUM_PAGES significantly will lead to many downloads.
NUM_PAGES = 10  # Adjust this value responsibly

def fetch_ebook_ids(num_pages):
    """
    Crawl the Gutenberg catalog to fetch eBook IDs.
    This function uses the search page sorted by downloads.
    """
    ebook_ids = set()
    base_url = "https://www.gutenberg.org/ebooks/search/?sort_order=downloads"
    for page in range(num_pages):
        # Gutenberg catalog pages use start_index parameter: page 1=1, page 2=26, page 3=51, etc.
        start_index = page * 25 + 1
        url = f"{base_url}&start_index={start_index}"
        print(f"Fetching catalog page: {url}")
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"Failed to fetch catalog page {url} (status {resp.status_code}).")
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            # Find all links with href of the form "/ebooks/ID"
            for link in soup.find_all("a", href=True):
                href = link["href"]
                if href.startswith("/ebooks/"):
                    try:
                        ebook_id = int(href.split("/")[2])
                        ebook_ids.add(ebook_id)
                    except (IndexError, ValueError):
                        continue
            print(f"Found {len(ebook_ids)} unique eBook IDs so far.")
            # Be respectful: pause between catalog page requests.
            time.sleep(2)
        except Exception as e:
            print(f"Error fetching catalog page {url}: {e}")
    return sorted(ebook_ids)

def construct_urls(ebook_id):
    """Return a list of candidate URLs for a given Gutenberg eBook ID."""
    return [
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-0.txt",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}.txt",
        f"https://www.gutenberg.org/ebooks/{ebook_id}.txt.utf-8",
        f"https://www.gutenberg.org/files/{ebook_id}/{ebook_id}-8.txt"
    ]

def download_ebook(ebook_id):
    """Attempt to download the eBook text from various URL patterns."""
    urls = construct_urls(ebook_id)
    for url in urls:
        try:
            print(f"Trying to download eBook {ebook_id} from {url} ...")
            response = requests.get(url, timeout=15)
            if response.status_code == 200 and len(response.content) > 1000:
                print(f"Downloaded eBook {ebook_id} successfully from {url}.")
                return response.text
            else:
                print(f"URL {url} returned status {response.status_code} or too little content.")
        except Exception as e:
            print(f"Error downloading from {url}: {e}")
    print(f"Failed to download eBook {ebook_id}.")
    return None

def main():
    output_file = "input.txt"
    all_text = ""

    # First, fetch a (limited) list of eBook IDs from the catalog.
    ebook_ids = fetch_ebook_ids(NUM_PAGES)
    print(f"Total unique eBook IDs found: {len(ebook_ids)}")

    # Download each eBook and append its text.
    for ebook_id in ebook_ids:
        text = download_ebook(ebook_id)
        if text:
            header = f"\n\n{'=' * 40}\nEBook ID: {ebook_id}\n{'=' * 40}\n\n"
            all_text += header + text
        # Pause between downloads to be respectful.
        time.sleep(2)

    # Write all downloaded texts to input.txt
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(all_text)
    print(f"All downloaded texts have been merged into {output_file}.")

if __name__ == "__main__":
    main()
