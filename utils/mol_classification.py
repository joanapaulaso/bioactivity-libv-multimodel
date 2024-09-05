import requests
import pandas as pd
import streamlit as st
import time
import os
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class TokenBucket:
    def __init__(self, tokens, fill_rate):
        self.capacity = tokens
        self.tokens = tokens
        self.fill_rate = fill_rate
        self.timestamp = time.time()

    def consume(self):
        now = time.time()
        tokens_to_add = (now - self.timestamp) * self.fill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.timestamp = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


def create_retry_session(
    retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 503, 504)
):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


session = create_retry_session()
token_bucket = TokenBucket(
    tokens=5, fill_rate=0.1
)  # 5 tokens, refill 1 token every 10 seconds
cache = {}


def get_entity_classification(inchikey, format="json"):
    if inchikey in cache:
        return cache[inchikey]

    url = f"http://classyfire.wishartlab.com/entities/{inchikey}.{format}"
    while not token_bucket.consume():
        time.sleep(1)
    try:
        response = session.get(url, headers={"accept": "application/json"})
        response.raise_for_status()
        result = response.json()
        cache[inchikey] = result
        time.sleep(2)  # Sleep for 2 seconds after each request
        return result
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching classification for {inchikey}: {str(e)}")
        return None


def submit_query(label, input_data, query_type="STRUCTURE"):
    url = "http://classyfire.wishartlab.com/queries"
    data = {"label": label, "query_input": input_data, "query_type": query_type}
    while not token_bucket.consume():
        time.sleep(1)
    try:
        response = session.post(
            url,
            json=data,
            headers={"accept": "application/json", "content-type": "application/json"},
        )
        response.raise_for_status()
        time.sleep(2)  # Sleep for 2 seconds after each request
        return response.json()
    except requests.exceptions.RequestException as e:
        if e.response is not None and e.response.status_code == 429:
            st.warning("Rate limit exceeded. Waiting before retrying...")
            time.sleep(60)  # Wait for 60 seconds before retrying
            return submit_query(label, input_data, query_type)  # Retry the request
        st.error(f"Error submitting query: {str(e)}")
        return None


def get_query_results(query_id, format="json"):
    url = f"http://classyfire.wishartlab.com/queries/{query_id}.{format}"
    while not token_bucket.consume():
        time.sleep(1)
    try:
        response = session.get(url, headers={"accept": "application/json"})
        response.raise_for_status()
        time.sleep(2)  # Sleep for 2 seconds after each request
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching query results: {str(e)}")
        return None


def classify_compounds_batch(molecules_processed, batch_size=5):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, len(molecules_processed), batch_size):
        batch = molecules_processed.iloc[i : i + batch_size]
        batch_results = classify_compound_batch(batch, status_text)
        results.extend(batch_results)
        progress = min((i + batch_size) / len(molecules_processed), 1.0)
        progress_bar.progress(progress)
        status_text.text(
            f"Processed {min(i+batch_size, len(molecules_processed))} out of {len(molecules_processed)} compounds"
        )
        time.sleep(10)  # Pause for 10 seconds between batches

    progress_bar.progress(1.0)
    status_text.text(f"Processed all {len(molecules_processed)} compounds")

    result_df = pd.concat(
        [molecules_processed, pd.DataFrame(results, index=molecules_processed.index)],
        axis=1,
    )

    # Create 'classes' folder if it doesn't exist
    if not os.path.exists("classes"):
        os.makedirs("classes")

    # Save the classified data as CSV
    result_df.to_csv("classes/classified_molecules.csv", index=False)
    st.success("Classified molecular data saved to 'classes/classified_molecules.csv'")

    return result_df


def classify_compound_batch(batch, status_text):
    try:
        classes = []
        for index, row in batch.iterrows():
            smiles = row["canonical_smiles"]
            inchi = row.get("inchi", "")  # Assuming you have InChI in your DataFrame

            if inchi:
                inchikey = inchi.split("=")[1].split("-")[0]
                classification = get_entity_classification(inchikey)
                if classification and "class" in classification:
                    class_name = classification["class"]["name"]
                    classes.append({"compound_class": class_name})
                    status_text.text(f"Classified {row.name}: {class_name}")
                    continue

            query = submit_query(f"Compound_{index}", smiles)
            if query and "id" in query:
                query_id = query["id"]

                wait_time = 5
                max_attempts = 5
                for attempt in range(max_attempts):
                    results = get_query_results(query_id)
                    if results and "classification_status" in results:
                        if results["classification_status"] == "Done":
                            if "entities" in results and len(results["entities"]) > 0:
                                class_name = results["entities"][0]["class"]["name"]
                                classes.append({"compound_class": class_name})
                                status_text.text(f"Classified {row.name}: {class_name}")
                            else:
                                classes.append({"compound_class": "Unknown"})
                                status_text.text(f"Classified {row.name}: Unknown")
                            break
                        elif results["classification_status"] == "In progress":
                            time.sleep(wait_time)
                            wait_time *= 2
                        else:
                            classes.append({"compound_class": "Error"})
                            status_text.text(f"Error classifying {row.name}")
                            break
                    else:
                        classes.append({"compound_class": "Error"})
                        status_text.text(f"Error classifying {row.name}")
                        break
                else:
                    classes.append({"compound_class": "Timeout"})
                    status_text.text(f"Timeout classifying {row.name}")
            else:
                classes.append({"compound_class": "Unknown"})
                status_text.text(f"Unable to classify {row.name}")

            time.sleep(2)  # Sleep for 2 seconds between compounds within a batch

        return classes
    except Exception as e:
        st.error(f"Error in compound classification: {str(e)}")
        return [{"compound_class": "Error"}] * len(batch)


# Load and save cache
def load_cache():
    try:
        with open("classyfire_cache.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_cache():
    with open("classyfire_cache.json", "w") as f:
        json.dump(cache, f)


# Main function to classify compounds
def classify_compound(molecules_processed=None):
    global cache
    cache = load_cache()

    st.subheader("Compound Classification")

    classification_option = st.radio(
        "Choose classification method:",
        ("Use existing data", "Upload pre-classified data", "Classify new data"),
    )

    st.write(f"Debug: Selected option - {classification_option}")

    if classification_option == "Upload pre-classified data":
        st.write("Debug: Entering upload section")
        uploaded_file = st.file_uploader("Upload pre-classified CSV file", type="csv")
        st.write(f"Debug: Uploaded file object - {uploaded_file}")

        if uploaded_file is not None:
            st.write("Debug: File uploaded successfully")
            try:
                result_df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.write(result_df.head())

                required_columns = ["canonical_smiles", "compound_class"]
                if all(col in result_df.columns for col in required_columns):
                    st.success("Pre-classified data loaded successfully.")

                    # Create 'classes' folder if it doesn't exist
                    if not os.path.exists("classes"):
                        os.makedirs("classes")

                    # Save the uploaded data to the 'classes' folder
                    result_df.to_csv("classes/classified_molecules.csv", index=False)
                    st.success(
                        "Uploaded data saved to 'classes/classified_molecules.csv'"
                    )

                    return result_df
                else:
                    missing_columns = [
                        col for col in required_columns if col not in result_df.columns
                    ]
                    st.error(
                        f"The uploaded file is missing required columns: {', '.join(missing_columns)}"
                    )
                    return None
            except Exception as e:
                st.error(f"Error reading the uploaded file: {str(e)}")
                return None
        else:
            st.info("Please upload a CSV file with pre-classified data.")
            return None

    elif classification_option == "Use existing data":
        st.write("Debug: Using existing data")
        if molecules_processed is not None and not molecules_processed.empty:
            result_df = classify_compounds_batch(molecules_processed)
            save_cache()
            return result_df
        else:
            st.error("No existing data available. Please choose another option.")
            return None

    elif classification_option == "Classify new data":
        st.write("Debug: Classifying new data")
        if molecules_processed is not None and not molecules_processed.empty:
            result_df = classify_compounds_batch(molecules_processed)
            save_cache()
            return result_df
        else:
            st.error(
                "No data available for classification. Please provide molecule data first."
            )
            return None

    return None
