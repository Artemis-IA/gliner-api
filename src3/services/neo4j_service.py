from neo4j import GraphDatabase, Transaction
from loguru import logger
from typing import Dict, Any, Optional

class Neo4jService:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed.")

    def create_node(self, label: str, properties: Dict[str, Any]) -> Optional[int]:
        with self.driver.session() as session:
            result = session.write_transaction(self._create_node_transaction, label, properties)
            return result

    @staticmethod
    def _create_node_transaction(tx: Transaction, label: str, properties: Dict[str, Any]) -> Optional[int]:
        query = f"""
        CREATE (n:{label} $properties)
        RETURN id(n) AS node_id
        """
        try:
            result = tx.run(query, properties=properties)
            node_id = result.single()["node_id"]
            logger.info(f"Node created with ID: {node_id}")
            return node_id
        except Exception as e:
            logger.error(f"Failed to create node: {e}")
            return None

    def create_relationship(self, source_id: int, target_id: int, relationship_type: str, properties: Dict[str, Any] = None) -> bool:
        with self.driver.session() as session:
            success = session.write_transaction(self._create_relationship_transaction, source_id, target_id, relationship_type, properties)
            return success

    @staticmethod
    def _create_relationship_transaction(tx: Transaction, source_id: int, target_id: int, relationship_type: str, properties: Dict[str, Any] = None) -> bool:
        query = f"""
        MATCH (a), (b)
        WHERE id(a) = $source_id AND id(b) = $target_id
        CREATE (a)-[r:{relationship_type} $properties]->(b)
        RETURN r
        """
        try:
            result = tx.run(query, source_id=source_id, target_id=target_id, properties=properties or {})
            if result.single():
                logger.info(f"Relationship {relationship_type} created between nodes {source_id} and {target_id}")
                return True
            else:
                logger.error("Failed to create relationship: No result returned")
                return False
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return False

    def get_node(self, node_id: int) -> Optional[Dict[str, Any]]:
        with self.driver.session() as session:
            node = session.read_transaction(self._get_node_transaction, node_id)
            return node

    @staticmethod
    def _get_node_transaction(tx: Transaction, node_id: int) -> Optional[Dict[str, Any]]:
        query = """
        MATCH (n)
        WHERE id(n) = $node_id
        RETURN properties(n) AS properties
        """
        try:
            result = tx.run(query, node_id=node_id)
            record = result.single()
            if record:
                logger.info(f"Node retrieved with ID: {node_id}")
                return record["properties"]
            else:
                logger.error(f"Node with ID {node_id} not found")
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve node: {e}")
            return None

    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> Any:
        with self.driver.session() as session:
            result = session.run(query, **(parameters or {}))
            return result.data()
