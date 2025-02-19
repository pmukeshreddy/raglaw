from transformers import pipeline, AutoTokenizer
from typing import List, Dict, Any, Optional

class DocumentSummarizer:
    """Class for summarizing documents using transformer-based models."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn",
                 max_input_tokens: int = 250,
                 max_output_tokens: int = 150,
                 min_output_tokens: int = 100) -> None:
        """
        Initialize the DocumentSummarizer with specified model and parameters.
        
        Args:
            model_name: Name of the pretrained model to use for summarization
            max_input_tokens: Maximum number of tokens for input text
            max_output_tokens: Maximum number of tokens for output summary
            min_output_tokens: Minimum number of tokens for output summary
        """
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.min_output_tokens = min_output_tokens
        
        # Initialize tokenizer and summarization pipeline
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.summarizer = pipeline("summarization", model=model_name)
    
    def summarize_text(self, text: str) -> str:
        """
        Summarize a single text document.
        
        Args:
            text: The input text to summarize
            
        Returns:
            A summary of the input text
        """
        # Tokenize and truncate if necessary
        tokenized = self.tokenizer.encode(text, truncation=True, 
                                          max_length=self.max_input_tokens)
        truncated_text = self.tokenizer.decode(tokenized)
        
        # Generate summary
        summary = self.summarizer(truncated_text, 
                                 max_length=self.max_output_tokens,
                                 min_length=self.min_output_tokens,
                                 do_sample=False)
        
        return summary[0]["summary_text"]
    
    def summarize_multiple(self, texts: List[str]) -> List[str]:
        """
        Summarize multiple documents individually.
        
        Args:
            texts: List of input texts to summarize
            
        Returns:
            List of summaries corresponding to input texts
        """
        return [self.summarize_text(doc) for doc in texts]
    
    def create_combined_summary(self, texts: List[str]) -> str:
        """
        Create individual summaries and then combine them.
        
        Args:
            texts: List of input texts to summarize
            
        Returns:
            Combined summary of all input texts
        """
        # Get individual summaries
        summaries = self.summarize_multiple(texts)
        
        # Combine them
        combined = " ".join(summaries)
        
        return combined


class SummaryProcessor:
    """Class for processing retrieved documents and their summaries."""
    
    def __init__(self, summarizer: DocumentSummarizer) -> None:
        """
        Initialize with a DocumentSummarizer instance.
        
        Args:
            summarizer: Instance of DocumentSummarizer to use for summarization
        """
        self.summarizer = summarizer
    
    def process_retrieved_docs(self, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process retrieved documents and generate summaries.
        
        Args:
            retrieved_docs: List of dictionaries containing document data
            
        Returns:
            Dictionary with individual and combined summaries
        """
        # Extract existing summaries if available
        chunks = [entry['summary'] for entry in retrieved_docs]
        
        # Create individual summaries if needed
        individual_summaries = self.summarizer.summarize_multiple(chunks)
        
        # Create combined summary
        combined_summary = self.summarizer.create_combined_summary(chunks)
        
        return {
            'individual_summaries': individual_summaries,
            'combined_summary': combined_summary,
            'original_docs': retrieved_docs
        }


