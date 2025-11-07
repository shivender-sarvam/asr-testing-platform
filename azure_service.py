import os
import csv
import io
from azure.storage.blob import BlobServiceClient, ContentSettings
from typing import Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_azure_config():
    """
    Get Azure storage configuration from environment variables or Streamlit secrets.
    Works in both Flask (os.environ) and Streamlit (st.secrets) contexts.
    """
    # Try Streamlit secrets first (for Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            account_name = st.secrets.get('AZURE_STORAGE_ACCOUNT_NAME', 'sarvamweb')
            container_name = st.secrets.get('AZURE_STORAGE_CONTAINER_NAME', 'whatsappmedia')
            account_key = st.secrets.get('AZURE_STORAGE_ACCOUNT_KEY')
            if account_key:
                return account_name, container_name, account_key
    except (ImportError, AttributeError, RuntimeError):
        pass  # Not in Streamlit context, fall back to os.environ
    
    # Fall back to environment variables (for Flask/Render)
    account_name = os.environ.get('AZURE_STORAGE_ACCOUNT_NAME', 'sarvamweb')
    container_name = os.environ.get('AZURE_STORAGE_CONTAINER_NAME', 'whatsappmedia')
    account_key = os.environ.get('AZURE_STORAGE_ACCOUNT_KEY')
    
    return account_name, container_name, account_key

def upload_csv_to_blob(csv_file_path: str, folder_name: str = "ASR Testing Dump", 
                      blob_filename: Optional[str] = None, 
                      add_timestamp: bool = False) -> str:
    """
    Uploads a CSV file to Azure Blob Storage in a specified folder.
    
    Args:
        csv_file_path (str): Path to the CSV file to upload
        folder_name (str): Folder name in the blob container (default: "ASR Testing Dump")
        blob_filename (str, optional): Custom filename for the blob. If None, uses original filename
        add_timestamp (bool): Whether to add timestamp to filename (default: False)
    
    Returns:
        str: The URL of the uploaded blob
    """
    # Azure Storage Configuration (works with both Flask env vars and Streamlit secrets)
    account_name, container_name, account_key = get_azure_config()
    
    if not account_key:
        raise ValueError("AZURE_STORAGE_ACCOUNT_KEY not set in environment variables or Streamlit secrets")
    
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    try:
        # Check if file exists
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file does not exist: {csv_file_path}")
        
        # Validate it's a CSV file
        if not csv_file_path.lower().endswith('.csv'):
            raise ValueError("File must have .csv extension")
        
        # Get filename
        if blob_filename is None:
            blob_filename = os.path.basename(csv_file_path)
        
        # Add timestamp if requested
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(blob_filename)
            blob_filename = f"{name}_{timestamp}{ext}"
        
        # Construct blob path with folder
        blob_path = f"{folder_name}/{blob_filename}"
        
        # Initialize Azure clients
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        
        # Read and upload CSV file
        logger.info(f"Uploading {csv_file_path} to {blob_path}")
        
        with open(csv_file_path, "rb") as data:
            blob_client.upload_blob(
                data,
                overwrite=True,
                content_settings=ContentSettings(content_type="text/csv")
            )
        
        # Generate URL
        url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_path}"
        
        logger.info(f"Upload successful! File uploaded to: {blob_path}")
        logger.info(f"URL: {url}")
        
        return url
        
    except Exception as e:
        logger.error(f"Error uploading CSV file: {e}")
        raise


def upload_csv_data_to_blob(csv_data: str, filename: str, 
                           folder_name: str = "ASR Testing Dump", 
                           add_timestamp: bool = False) -> str:
    """
    Uploads CSV data as string to Azure Blob Storage.
    
    Args:
        csv_data (str): CSV data as string
        filename (str): Filename for the CSV (should include .csv extension)
        folder_name (str): Folder name in the blob container (default: "ASR Testing Dump")
        add_timestamp (bool): Whether to add timestamp to filename (default: False)
    
    Returns:
        str: The URL of the uploaded blob
    """
    # Azure Storage Configuration (works with both Flask env vars and Streamlit secrets)
    account_name, container_name, account_key = get_azure_config()
    
    if not account_key:
        raise ValueError("AZURE_STORAGE_ACCOUNT_KEY not set in environment variables or Streamlit secrets")
    
    connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
    
    try:
        # Ensure filename has .csv extension
        if not filename.lower().endswith('.csv'):
            filename += '.csv'
        
        # Add timestamp if requested
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(filename)
            filename = f"{name}_{timestamp}{ext}"
        
        # Construct blob path with folder
        blob_path = f"{folder_name}/{filename}"
        
        # Initialize Azure clients
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        blob_client = container_client.get_blob_client(blob_path)
        
        # Upload CSV data
        logger.info(f"Uploading CSV data to {blob_path}")
        
        blob_client.upload_blob(
            csv_data,
            overwrite=True,
            content_settings=ContentSettings(content_type="text/csv")
        )
        
        # Generate URL
        url = f"https://{account_name}.blob.core.windows.net/{container_name}/{blob_path}"
        
        logger.info(f"Upload successful! CSV data uploaded to: {blob_path}")
        logger.info(f"URL: {url}")
        
        return url
        
    except Exception as e:
        logger.error(f"Error uploading CSV data: {e}")
        raise


# ASR Testing specific functions
def upload_single_test_result(test_result: dict, user_email: str, language: str, session_id: str) -> str:
    """
    Upload a single ASR test result to Azure Blob Storage immediately.
    This ensures results are saved even if session is lost.
    
    Args:
        test_result (dict): Single test result dictionary
        user_email (str): User's email address
        language (str): Test language
        session_id (str): Session identifier
    
    Returns:
        str: URL of uploaded CSV file
    """
    try:
        # Create CSV data for single result
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['user_email', 'language', 'session_id', 'crop_name', 'attempt_number', 'transcript', 'keyword_detected', 'timestamp', 'upload_timestamp'])
        
        # Write single result
        upload_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([
            user_email,
            language,
            session_id,
            test_result.get('crop_name', ''),
            test_result.get('attempt_number', ''),
            test_result.get('transcript', ''),
            test_result.get('keyword_detected', ''),
            test_result.get('timestamp', ''),
            upload_timestamp
        ])
        
        csv_data = output.getvalue()
        output.close()
        
        # Generate filename for this session
        filename = f"asr_test_results_{user_email}_{language}_{session_id}.csv"
        
        # Upload to Azure
        url = upload_csv_data_to_blob(
            csv_data=csv_data,
            filename=filename,
            folder_name="ASR Testing Dump",
            add_timestamp=False
        )
        
        logger.info(f"Single ASR test result uploaded successfully: {url}")
        return url
        
    except Exception as e:
        logger.error(f"Error uploading single ASR test result: {e}")
        raise

def recover_session_from_azure(user_email: str, language: str, session_id: str) -> list:
    """
    Recover session data from Azure when session is lost.
    This is a recovery mechanism for lost session data.
    
    Args:
        user_email (str): User's email address
        language (str): Test language
        session_id (str): Session identifier
    
    Returns:
        list: List of recovered test results
    """
    try:
        # Azure Storage Configuration
        account_name = os.environ.get('AZURE_STORAGE_ACCOUNT_NAME', 'sarvamweb')
        container_name = os.environ.get('AZURE_STORAGE_CONTAINER_NAME', 'whatsappmedia')
        account_key = os.environ.get('AZURE_STORAGE_ACCOUNT_KEY')
        
        if not account_key:
            logger.error("AZURE_STORAGE_ACCOUNT_KEY not set - cannot recover from Azure")
            return []
        
        connection_string = f"DefaultEndpointsProtocol=https;AccountName={account_name};AccountKey={account_key};EndpointSuffix=core.windows.net"
        
        # Initialize Azure clients
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)
        
        # List blobs for this session
        blob_prefix = f"ASR Testing Dump/asr_test_results_{user_email}_{language}_{session_id}"
        blobs = container_client.list_blobs(name_starts_with=blob_prefix)
        
        recovered_results = []
        for blob in blobs:
            try:
                # Download and parse CSV
                blob_client = container_client.get_blob_client(blob.name)
                csv_data = blob_client.download_blob().readall().decode('utf-8')
                
                # Parse CSV data
                csv_reader = csv.DictReader(io.StringIO(csv_data))
                for row in csv_reader:
                    if row.get('session_id') == session_id:
                        recovered_results.append({
                            'crop_name': row.get('crop_name', ''),
                            'attempt_number': int(row.get('attempt_number', 1)),
                            'transcript': row.get('transcript', ''),
                            'keyword_detected': row.get('keyword_detected', '').lower() == 'true',
                            'timestamp': row.get('timestamp', '')
                        })
                
                logger.info(f"Recovered {len(recovered_results)} results from Azure blob: {blob.name}")
                
            except Exception as e:
                logger.error(f"Error recovering from blob {blob.name}: {e}")
                continue
        
        logger.info(f"Total recovered results: {len(recovered_results)}")
        return recovered_results
        
    except Exception as e:
        logger.error(f"Error recovering session from Azure: {e}")
        return []

def upload_asr_test_results(test_results: list, user_email: str, language: str, session_id: str) -> str:
    """
    Upload ASR test results to Azure Blob Storage.
    
    Args:
        test_results (list): List of test result dictionaries
        user_email (str): User's email address
        language (str): Test language
        session_id (str): Session identifier
    
    Returns:
        str: URL of uploaded CSV file
    """
    try:
        # Create CSV data manually
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['user_email', 'language', 'session_id', 'crop_name', 'attempt_number', 'transcript', 'keyword_detected', 'timestamp', 'upload_timestamp'])
        
        # Write data
        upload_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for result in test_results:
            writer.writerow([
                user_email,
                language,
                session_id,
                result.get('crop_name', ''),
                result.get('attempt_number', ''),
                result.get('transcript', ''),
                result.get('keyword_detected', ''),
                result.get('timestamp', ''),
                upload_timestamp
            ])
        
        csv_data = output.getvalue()
        output.close()
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"asr_test_results_{user_email}_{language}_{timestamp}.csv"
        
        # Upload to Azure
        url = upload_csv_data_to_blob(
            csv_data=csv_data,
            filename=filename,
            folder_name="ASR Testing Dump",
            add_timestamp=False  # We already added timestamp to filename
        )
        
        logger.info(f"ASR test results uploaded successfully: {url}")
        return url
        
    except Exception as e:
        logger.error(f"Error uploading ASR test results: {e}")
        raise

