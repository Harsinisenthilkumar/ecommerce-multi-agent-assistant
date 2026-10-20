from threading import Lock
from uuid import uuid4

from langchain_core.tools import tool

PRODUCT_CATALOG = {
    "sku101": {"name": "Wireless Gaming Mouse", "stock": 15, "price": "$49.99"},
    "sku102": {"name": "Mechanical Keyboard", "stock": 8, "price": "$89.99"},
    "sku103": {"name": "HD Monitor 27-inch", "stock": 4, "price": "$229.99"}
}

ORDER_DATABASE = {
    "ord501": {"status": "Shipped", "delivery_date": "Oct 24, 2026", "item": "Wireless Gaming Mouse"},
    "ord502": {"status": "Processing", "delivery_date": "Oct 28, 2026", "item": "Mechanical Keyboard"}
}

_database_lock = Lock()


@tool
def check_stock(sku: str) -> str:
    """Check inventory and stock levels for a given product SKU."""
    sku_clean = sku.lower().strip()
    with _database_lock:
        item = PRODUCT_CATALOG.get(sku_clean)
        if item is not None:
            item = dict(item)
    if item is not None:
        return f"Product: {item['name']}, Stock: {item['stock']} units available, Price: {item['price']}"
    return f"Sorry, SKU {sku} was not found in our catalog."


@tool
def create_order(sku: str, quantity: int, address: str) -> str:
    """Create an order in the mock database and reserve its inventory."""
    sku_clean = sku.lower().strip()
    if isinstance(quantity, bool) or quantity <= 0:
        raise ValueError("quantity must be a positive integer.")
    if not address.strip():
        raise ValueError("address must be a non-empty string.")

    with _database_lock:
        item = PRODUCT_CATALOG.get(sku_clean)
        if item is None:
            return f"Order not created: SKU {sku} was not found in our catalog."
        if item["stock"] < quantity:
            return (
                f"Order not created: only {item['stock']} unit(s) of {sku} "
                f"are available; {quantity} requested."
            )

        item["stock"] -= quantity
        order_id = f"ord{uuid4().hex[:6]}"
        ORDER_DATABASE[order_id] = {
            "status": "Processing",
            "delivery_date": "To be confirmed",
            "item": item["name"],
            "quantity": quantity,
            "address": address.strip(),
        }

    return f"Order created successfully. Order ID: {order_id}; status: Processing."


@tool
def check_order_status(order_id: str) -> str:
    """Check shipping and delivery status for a given order ID."""
    oid_clean = order_id.lower().strip()
    with _database_lock:
        order = ORDER_DATABASE.get(oid_clean)
        if order is not None:
            order = dict(order)
    if order is not None:
        return f"Order ID {oid_clean} ({order['item']}) status is '{order['status']}', expected delivery by {order['delivery_date']}."
    return f"Sorry, order ID {order_id} could not be found."
