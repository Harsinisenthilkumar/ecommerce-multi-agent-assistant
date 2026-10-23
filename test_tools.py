import unittest

from agents.tools import check_inventory_stock


class InventoryToolTests(unittest.TestCase):
    def test_check_inventory_accepts_numeric_sku_variants(self):
        for sku in ("101", "SKU 101", "SKU101", "SKU-101"):
            with self.subTest(sku=sku):
                result = check_inventory_stock.invoke({"sku": sku})
                self.assertEqual(
                    result,
                    "SKU 'SKU-101' is in stock: 15 unit(s) available.",
                )


if __name__ == "__main__":
    unittest.main()
