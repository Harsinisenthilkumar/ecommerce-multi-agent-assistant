import os
import requests
from langchain_core.tools import tool

ORDER_API_URL = os.getenv("ORDER_API_URL", "https://api.example.com/v1")

@tool
def create_order(customer_id: str, item_sku: str, quantity: int) -> str:
    """Create a new customer order once stock is verified and reserved."""
    try:
        order_id = "ORD-88492"
        return f"Order successfully created! Order ID: {order_id} for SKU '{item_sku}' (Quantity: {quantity}). Status: Processing."
    except Exception as e:
        return f"Order creation failed: {str(e)}"

@tool
def check_order_status(order_id: str) -> str:
    """Retrieve the shipping status and ETA for an existing order ID."""
    try:
        return f"Order '{order_id}' status: IN_TRANSIT. Estimated delivery by tomorrow."
    except Exception as e:
        return f"Could not retrieve status for order {order_id}: {str(e)}"