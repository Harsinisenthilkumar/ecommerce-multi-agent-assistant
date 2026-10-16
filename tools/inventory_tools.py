import os
import requests
from langchain_core.tools import tool

INVENTORY_API_URL = os.getenv("INVENTORY_API_URL", "https://api.example.com/v1")

@tool
def check_inventory_stock(item_sku: str) -> str:
    """Check the current available stock quantity for a given product SKU."""
    try:
        mock_stock_db = {
            "SKU-101": 15,
            "SKU-102": 0,
            "SKU-103": 8
        }
        stock = mock_stock_db.get(item_sku.upper(), 5)
        
        if stock > 0:
            return f"Success: Item '{item_sku}' is IN STOCK with {stock} units available."
        return f"Out of stock: Item '{item_sku}' is currently unavailable."
    except Exception as e:
        return f"Error connecting to inventory service: {str(e)}"

@tool
def reserve_inventory_stock(item_sku: str, quantity: int) -> str:
    """Reserve a specified quantity of stock for an active order workflow."""
    try:
        return f"Success: Temporarily reserved {quantity} unit(s) of SKU '{item_sku}'."
    except Exception as e:
        return f"Failed to reserve stock for {item_sku}: {str(e)}"