# smart_research_assistant/langsmith_module.py

import os
import uuid
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tracers import LangChainTracer
from langchain_core.callbacks import CallbackManager

# Import our modules
from smart_research_assistant.langchain_module import ResearchAssistantLangChain
from smart_research_assistant.langgraph_module import ResearchAssistantLangGraph

# Load environment variables
load_dotenv()

# Simple fallback mock for langsmith client
class MockLangSmithClient:
    def __init__(self, *args, **kwargs):
        pass
    
    def create_dataset(self, *args, **kwargs):
        return "mock-dataset"
    
    def create_example(self, *args, **kwargs):
        return "mock-example"
    
    def create_feedback(self, *args, **kwargs):
        return "mock-feedback"
    
    def list_runs(self, *args, **kwargs):
        return []

# Initialize LangSmith client - with error handling for missing keys
try:
    # Try to import the actual client
    from langsmith import Client as LangSmithClient
    langsmith_client = LangSmithClient(
        api_key=os.getenv("LANGCHAIN_API_KEY"),
        api_url=os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"),
    )

except Exception as e:
    print(f"Warning: LangSmith client initialization failed: {e}")
    print("Using mock LangSmith client instead")
    langsmith_client = MockLangSmithClient()





class ResearchAssistantLangSmith:

    
    def __init__(self):
        """Initialize the Research Assistant with LangSmith tracing"""
        # Create callback manager with LangSmith tracer if available
        try:
            self.tracer = LangChainTracer(
                project_name="smart_research_assistant"
            )
            self.callback_manager = CallbackManager([self.tracer])
            
            # Initialize LangChain module with tracing
            self.langchain_module = ResearchAssistantLangChain()
            self.langchain_module.llm = ChatOpenAI(
                callbacks=[self.tracer]
            )
            
            # Initialize LangGraph module
            self.langgraph_module = ResearchAssistantLangGraph()
            
            self.tracing_enabled = True
        except Exception as e:
            print(f"Warning: LangSmith tracing disabled: {e}")
            self.langchain_module = ResearchAssistantLangChain()
            self.langgraph_module = ResearchAssistantLangGraph()
            self.tracing_enabled = False
    
    def execute_with_tracing(self, query: str) -> Dict[str, Any]:

        # Check if tracing is enabled
        if not self.tracing_enabled:
            # Fall back to regular execution without tracing
            result = self.langgraph_module.execute_research(query)
            return {"result": result}
        
        # Create a new run
        run_id = str(uuid.uuid4())
        
        try:
            # Start the run
            with self.tracer.start_trace(name="research_query", run_id=run_id):
                # Execute the research
                result = self.langgraph_module.execute_research(query)
                
                # Log the result
                self.tracer.on_chain_end(
                    outputs={"result": result},
                    run_id=run_id
                )
            
            # Return the result with the run ID for later reference
            return {
                "result": result,
                "run_id": run_id,
                "project": "smart_research_assistant"
            }
        except Exception as e:
            print(f"Tracing error: {e}")
            # Fall back to regular execution
            result = self.langgraph_module.execute_research(query)
            return {"result": result}
    
    def log_feedback(self, run_id: str, feedback_type: str, score: float, comment: Optional[str] = None):
        """
        Log feedback for a specific run
        
        Args:
            run_id: ID of the run to provide feedback for
            feedback_type: Type of feedback (relevance, comprehensiveness, etc.)
            score: Score from 1-10
            comment: Optional comment
        """
        if not self.tracing_enabled:
            print(f"Feedback logged (mock): {feedback_type}={score}")
            return
            
        try:
            langsmith_client.create_feedback(
                run_id=run_id,
                key=feedback_type,
                score=score,
                comment=comment
            )
            print(f"Feedback logged: {feedback_type}={score}")
        except Exception as e:
            print(f"Error logging feedback: {e}")
    
    def analyze_performance(self, project_name: str = "smart_research_assistant") -> Dict[str, Any]:
        """
        Analyze the performance of the research assistant
        
        Args:
            project_name: Name of the project to analyze
            
        Returns:
            Performance metrics
        """
        if not self.tracing_enabled:
            return {
                "message": "LangSmith tracing is disabled. Performance metrics not available.",
                "total_runs": 0,
                "error_rate": 0,
                "avg_latency_seconds": 0,
                "project_name": project_name
            }
            
        try:
            # Get runs from the project
            runs = langsmith_client.list_runs(
                project_name=project_name,
                execution_order=1
            )
            
            # Calculate metrics
            total_runs = 0
            error_runs = 0
            avg_latency = 0
            
            for run in runs:
                total_runs += 1
                if getattr(run, 'error', None):
                    error_runs += 1
                
                # Handle different API versions
                start_time = getattr(run, 'start_time', None)
                end_time = getattr(run, 'end_time', None)
                
                if end_time and start_time:
                    try:
                        # Handle different time formats
                        if isinstance(start_time, str) and isinstance(end_time, str):
                            from datetime import datetime
                            start = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                            end = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                            avg_latency += (end - start).total_seconds()
                        else:
                            avg_latency += (end_time - start_time).total_seconds()
                    except Exception as e:
                        print(f"Error calculating latency: {e}")
            
            if total_runs > 0:
                avg_latency /= total_runs
            
            # Return metrics
            return {
                "total_runs": total_runs,
                "error_rate": error_runs / total_runs if total_runs > 0 else 0,
                "avg_latency_seconds": avg_latency,
                "project_name": project_name
            }
        except Exception as e:
            return {
                "error": f"Error analyzing performance: {str(e)}",
                "total_runs": 0,
                "error_rate": 0,
                "avg_latency_seconds": 0,
                "project_name": project_name
            }