from threading import Lock
from uuid import uuid4

from langchain_core.tools import tool

_inventory = {
    "SKU-101": 15,
    "SKU-102": 0,
    "SKU-103": 8,
}
_orders: dict[str, dict[str, str | int]] = {
    "ORD-88492": {
        "sku": "SKU-101",
        "quantity": 2,
        "status": "IN_TRANSIT",
    }
}
_store_lock = Lock()


def _normalize_sku(sku: str) -> str:
    if not isinstance(sku, str) or not sku.strip():
        raise ValueError("sku must be a non-empty string.")
    return sku.strip().upper()


@tool
def check_inventory_stock(sku: str) -> str:
    """Check available stock for a product SKU."""
    normalized_sku = _normalize_sku(sku)
    with _store_lock:
        available = _inventory.get(normalized_sku)

    if available is None:
        return f"No inventory record was found for SKU '{normalized_sku}'."
    if available == 0:
        return f"SKU '{normalized_sku}' is out of stock."
    return f"SKU '{normalized_sku}' is in stock: {available} unit(s) available."


@tool
def create_order(sku: str, quantity: int, address: str) -> str:
    """Create a demo order, decrementing available stock for the requested SKU."""
    normalized_sku = _normalize_sku(sku)
    if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
        raise ValueError("quantity must be a positive integer.")
    if not isinstance(address, str) or not address.strip():
        raise ValueError("address must be a non-empty string.")

    with _store_lock:
        available = _inventory.get(normalized_sku)
        if available is None:
            return f"Order not created: no inventory record exists for SKU '{normalized_sku}'."
        if available < quantity:
            return (
                f"Order not created: SKU '{normalized_sku}' has only "
                f"{available} unit(s) available; {quantity} requested."
            )

        order_id = f"ORD-{uuid4().hex[:8].upper()}"
        _inventory[normalized_sku] = available - quantity
        _orders[order_id] = {
            "sku": normalized_sku,
            "quantity": quantity,
            "address": address.strip(),
            "status": "PROCESSING",
        }

    return (
        f"Order created successfully. Order ID: {order_id}; SKU: "
        f"{normalized_sku}; quantity: {quantity}; status: PROCESSING."
    )


@tool
def track_order(order_id: str) -> str:
    """Look up the current status of an order by its order ID."""
    if not isinstance(order_id, str) or not order_id.strip():
        raise ValueError("order_id must be a non-empty string.")
    normalized_order_id = order_id.strip().upper()

    with _store_lock:
        order = _orders.get(normalized_order_id)
        if order is None:
            return f"No order was found with ID '{normalized_order_id}'."
        order_details = dict(order)

    return (
        f"Order '{normalized_order_id}' status: {order_details['status']}; "
        f"SKU: {order_details['sku']}; quantity: {order_details['quantity']}."
    )



