import os
import logging
from neo4j import GraphDatabase
from dotenv import load_dotenv
import pickle
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import required libraries with error handling
try:
    import networkx as nx # type: ignore
    logger.info("Successfully imported networkx")
except ImportError as e:
    logger.error(f"Failed to import networkx: {e}")
    nx = None

try:
    import numpy as np
    logger.info("Successfully imported numpy")
except ImportError as e:
    logger.error(f"Failed to import numpy: {e}")
    np = None

try:
    from node2vec import Node2Vec # type: ignore
    logger.info("Successfully imported node2vec")
except ImportError as e:
    logger.error(f"Failed to import node2vec: {e}")
    Node2Vec = None

# Check if all required libraries are available
if nx is None or np is None or Node2Vec is None:
    logger.error("One or more required libraries are missing. Node2VecService will not function properly.")
    raise ImportError("Required libraries for Node2VecService are missing.")

load_dotenv()

class Node2VecService:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"), 
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
        )
        self.model = None
        self.embeddings = {}
        self.last_trained = None
        self.model_path = "node2vec_model.pkl"
        
    def load_or_train_model(self, force_retrain=False):
        """Load existing model or train a new one if needed"""
        try:
            # Check if we have a recent model file
            if not force_retrain and os.path.exists(self.model_path):
                file_age = time.time() - os.path.getmtime(self.model_path)
                # If model is less than 24 hours old, load it
                if file_age < 86400:  # 24 hours in seconds
                    logger.info("📚 Loading existing Node2Vec model...")
                    with open(self.model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    self.load_embeddings()
                    logger.info("✅ Node2Vec model loaded successfully")
                    return True
            
            logger.info("🔄 Training new Node2Vec model...")
            return self.train_model()
            
        except Exception as e:
            logger.error(f"❌ Error with Node2Vec model: {e}", exc_info=True)
            return False
    
    def train_model(self, align_embeddings=True):
        """Train Node2Vec model on the knowledge graph with optional alignment"""
        try:
            # Get graph from Neo4j
            G = self.get_graph_from_neo4j()
            
            if len(G.nodes()) == 0:
                logger.warning("⚠️ No nodes found in the graph for Node2Vec training")
                return False
            
            logger.info(f"📊 Training Node2Vec on {len(G.nodes())} nodes and {len(G.edges())} edges...")
            
            # Configure Node2Vec with enhanced parameters
            node2vec = Node2Vec(
                G,
                dimensions=128,    # Size of embedding vectors
                walk_length=40,    # Increased walk length for better context
                num_walks=300,     # Increased walks for better training
                workers=4,         # Number of CPU cores
                p=1.0,             # Return parameter
                q=1.0,             # In-out parameter
                weight_key='weight' if any('weight' in d for _, _, d in G.edges(data=True)) else None,
                quiet=True         # Reduce output noise
            )
            
            # Train model
            self.model = node2vec.fit(window=15, min_count=1, batch_words=4, epochs=5)
            
            # Store embeddings
            for node in G.nodes():
                if node in self.model.wv:
                    self.embeddings[str(node)] = self.model.wv[node]
            
            # Apply embedding alignment if requested
            if align_embeddings:
                logger.info("🔄 Applying embedding alignment...")
                self.align_embeddings()
                
                # Enhance embeddings with graph context
                logger.info("🔄 Enhancing embeddings with graph context...")
                self.enhance_embeddings_with_context()
            
            # Save model for future use
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            
            self.last_trained = time.time()
            logger.info(f"✅ Node2Vec model trained successfully on {len(self.embeddings)} nodes")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training Node2Vec model: {e}", exc_info=True)
            return False
    
    def get_graph_from_neo4j(self):
        """Extract graph from Neo4j for Node2Vec training"""
        G = nx.Graph()
        
        with self.driver.session() as session:
            # Skip UID assignment for now and use elementId directly
            # This avoids complex queries that might not work in all Neo4j versions
            pass
            
            # Get all nodes with their labels and properties
            result = session.run("""
                MATCH (n)
                RETURN elementId(n) as node_id, labels(n) as labels, properties(n) as properties
            """)
            
            # Store first label for relationship mapping
            first_label = None
            
            for record in result:
                element_id = record["node_id"]
                labels = record["labels"]
                properties = record["properties"]
                
                # Store first label for relationship mapping
                if first_label is None and labels:
                    first_label = labels[0]
                
                # Create node with elementId as identifier for stability
                node_identifier = f"{labels[0]}_{element_id}"
                G.add_node(node_identifier, element_id=element_id, **properties)
            
            # Get all relationships with proper mapping and edge properties
            result = session.run("""
                MATCH (s)-[r]->(t)
                RETURN elementId(s) as source_id, elementId(t) as target_id, 
                       type(r) as rel_type, properties(r) as properties,
                       elementId(r) as rel_id
            """)
            
            edge_count = 0
            for record in result:
                source_element_id = record["source_id"]
                target_element_id = record["target_id"]
                rel_type = record["rel_type"]
                properties = record["properties"]
                rel_id = record["rel_id"]
                
                # Map elementIds to our node identifiers
                source_node = f"{first_label}_{source_element_id}" if first_label else f"node_{source_element_id}"
                target_node = f"{first_label}_{target_element_id}" if first_label else f"node_{target_element_id}"
                
                # Find actual nodes by elementId
                actual_source_node = None
                actual_target_node = None
                
                for node in G.nodes():
                    if G.nodes[node].get('element_id') == source_element_id:
                        actual_source_node = node
                    if G.nodes[node].get('element_id') == target_element_id:
                        actual_target_node = node
                
                if actual_source_node and actual_target_node:
                    # Add edge with relationship type as weight and properties
                    edge_properties = {
                        'type': rel_type,
                        'element_id': rel_id,
                        **properties
                    }
                    G.add_edge(actual_source_node, actual_target_node, **edge_properties)
                    edge_count += 1
            
            print(f"📊 Extracted {len(G.nodes())} nodes and {edge_count} edges for Node2Vec training")
        
        return G
    
    def find_similar_edges(self, query_text, top_k=5):
        """Find edges similar to the query text using Node2Vec embeddings"""
        if not self.model or not self.embeddings:
            print("⚠️ Node2Vec model not available")
            return []
        
        try:
            # Create a virtual query node based on keywords
            query_terms = query_text.lower().split()
            
            # Get all edges from the graph and create node-edge-node units
            candidate_edges = []
            
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (s)-[r]->(t)
                    RETURN elementId(s) as source_id, elementId(t) as target_id, 
                           type(r) as rel_type, properties(r) as properties,
                           elementId(r) as rel_id, s.name as source_name, t.name as target_name
                """)
                
                for record in result:
                    source_element_id = record["source_id"]
                    target_element_id = record["target_id"]
                    rel_type = record["rel_type"]
                    properties = record["properties"]
                    rel_id = record["rel_id"]
                    source_name = record["source_name"]
                    target_name = record["target_name"]
                    
                    # Create edge text for matching
                    edge_text = f"{rel_type} {source_name} {target_name}".lower()
                    
                    # Check if query terms match the edge
                    if any(term in edge_text for term in query_terms):
                        # Find corresponding node identifiers
                        source_node = None
                        target_node = None
                        
                        for node_id in self.embeddings.keys():
                            if "_" in node_id:
                                _, element_id = node_id.split("_", 1)
                                try:
                                    element_id = int(element_id)
                                except ValueError:
                                    element_id = element_id
                                
                                if element_id == source_element_id:
                                    source_node = node_id
                                elif element_id == target_element_id:
                                    target_node = node_id
                        
                        if source_node and target_node:
                            # Create node-edge-node unit embedding
                            if source_node in self.embeddings and target_node in self.embeddings:
                                source_embedding = self.embeddings[source_node]
                                target_embedding = self.embeddings[target_node]
                                
                                # Create combined embedding: concatenate source, edge info, target
                                # Use relationship type as part of the embedding
                                rel_vector = self.create_relation_vector(rel_type, len(source_embedding))
                                
                                # Combine embeddings: source + relation + target
                                combined_embedding = np.concatenate([
                                    source_embedding,
                                    rel_vector,
                                    target_embedding
                                ])
                                
                                candidate_edges.append({
                                    'source': source_node,
                                    'target': target_node,
                                    'embedding': combined_embedding,
                                    'data': {
                                        'type': rel_type,
                                        'element_id': rel_id,
                                        **properties,
                                        'source_name': source_name,
                                        'target_name': target_name
                                    }
                                })
            
            if not candidate_edges:
                print("⚠️ No edges found containing query terms")
                return []
            
            # Create query embedding based on matching edges
            query_embedding = self.create_query_embedding_for_edges(query_terms, candidate_edges)
            
            if query_embedding is None:
                return []
            
            # Calculate cosine similarities
            similarities = []
            for edge in candidate_edges:
                similarity = self.cosine_similarity(query_embedding, edge['embedding'])
                similarities.append((edge['source'], edge['target'], similarity, edge['data']))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[2], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            print(f"❌ Error finding similar edges: {e}")
            return []
    
    def create_relation_vector(self, rel_type, embedding_dim):
        """Create a relationship vector based on the relationship type"""
        try:
            # Create a hash-based vector for the relationship type
            # Use a simple hash function to create a consistent vector
            rel_hash = hash(rel_type) % (2**32)
            
            # Create a vector with the same dimension as node embeddings
            rel_vector = np.zeros(embedding_dim)
            
            # Use the hash to set values in the vector
            for i in range(embedding_dim):
                rel_vector[i] = (rel_hash >> i) & 1
            
            # Normalize the vector
            norm = np.linalg.norm(rel_vector)
            if norm > 0:
                rel_vector = rel_vector / norm
            
            return rel_vector
            
        except Exception as e:
            print(f"❌ Error creating relation vector: {e}")
            return np.zeros(embedding_dim)
    
    def create_query_embedding_for_edges(self, query_terms, candidate_edges):
        """Create a query embedding based on matching node-edge-node units"""
        try:
            if not candidate_edges:
                return None
            
            # Average the embeddings of all matching node-edge-node units
            relevant_embeddings = [edge['embedding'] for edge in candidate_edges]
            
            # Calculate mean embedding
            query_embedding = np.mean(relevant_embeddings, axis=0)
            return query_embedding
            
        except Exception as e:
            print(f"❌ Error creating query embedding for edges: {e}")
            return None
    
    def find_similar_nodes(self, query_text, top_k=5):
        """Find nodes similar to the query text using Node2Vec embeddings"""
        if not self.model or not self.embeddings:
            print("⚠️ Node2Vec model not available")
            return []
        
        try:
            # Create a virtual query node based on keywords
            query_terms = query_text.lower().split()
            
            # Find nodes that contain query terms in their properties
            candidate_nodes = []
            for node_id in self.embeddings.keys():
                # Extract the original node properties from the identifier
                # Format: "Label_elementId"
                if "_" in node_id:
                    label_part = node_id.split("_")[0].lower()
                    # Check if query terms match the label
                    if any(term in label_part for term in query_terms):
                        candidate_nodes.append(node_id)
                        continue
                
                # Also check node properties if available
                # This would require additional property storage
                node_text = node_id.lower()
                if any(term in node_text for term in query_terms):
                    candidate_nodes.append(node_id)
            
            if not candidate_nodes:
                print("⚠️ No nodes found containing query terms")
                return []
            
            # Calculate similarity between query and candidate nodes
            query_embedding = self.create_query_embedding(query_terms, candidate_nodes)
            
            if query_embedding is None:
                return []
            
            # Calculate cosine similarities
            similarities = []
            for node_id in candidate_nodes:
                node_embedding = self.embeddings[node_id]
                similarity = self.cosine_similarity(query_embedding, node_embedding)
                similarities.append((node_id, similarity))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            print(f"❌ Error finding similar nodes: {e}")
            return []
    
    def create_query_embedding(self, query_terms, candidate_nodes):
        """Create an embedding for the query based on similar nodes"""
        try:
            # Average the embeddings of nodes that contain query terms
            relevant_embeddings = []
            
            for node_id in candidate_nodes:
                if node_id in self.embeddings:
                    relevant_embeddings.append(self.embeddings[node_id])
            
            if not relevant_embeddings:
                return None
            
            # Calculate mean embedding
            query_embedding = np.mean(relevant_embeddings, axis=0)
            return query_embedding
            
        except Exception as e:
            print(f"❌ Error creating query embedding: {e}")
            return None
    
    def cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            print(f"❌ Error calculating cosine similarity: {e}")
            return 0.0
    
    def get_node_context(self, node_id, max_depth=2):
        """Get context for a specific node including its neighbors"""
        try:
            with self.driver.session() as session:
                # Get node details using elementId
                node_result = session.run("""
                    MATCH (n)
                    WHERE elementId(n) = $node_id
                    RETURN n
                """, node_id=node_id)
                
                node = node_result.single()
                if not node:
                    return None
                
                # Get neighbors within specified depth
                neighbors_result = session.run("""
                    MATCH (n)-[*1..$max_depth]-(m)
                    WHERE elementId(n) = $node_id
                    RETURN DISTINCT m, relationships(n)-[]->(m) as rels
                """, node_id=node_id, max_depth=max_depth)
                
                context = {
                    "central_node": node["n"],
                    "neighbors": []
                }
                
                for record in neighbors_result:
                    neighbor_info = {
                        "node": record["m"],
                        "relationships": [rel for rel in record["rels"]]
                    }
                    context["neighbors"].append(neighbor_info)
                
                return context
                
        except Exception as e:
            print(f"❌ Error getting node context: {e}")
            return None
    
    def align_embeddings(self):
        """Apply embedding alignment to improve semantic coherence"""
        try:
            if not self.embeddings:
                print("⚠️ No embeddings to align")
                return
            
            print("🔄 Applying PCA-based alignment...")
            
            # Convert embeddings to numpy array
            node_ids = list(self.embeddings.keys())
            embedding_matrix = np.array([self.embeddings[node_id] for node_id in node_ids])
            
            # Apply PCA for dimensionality reduction and alignment
            try:
                from sklearn.decomposition import PCA
                
                # Reduce to 95% of variance while maintaining structure
                pca = PCA(n_components=0.95, whiten=True)
                aligned_embeddings = pca.fit_transform(embedding_matrix)
                
                # Update embeddings with aligned versions
                for i, node_id in enumerate(node_ids):
                    self.embeddings[node_id] = aligned_embeddings[i]
                
                print(f"✅ Embeddings aligned using PCA ({pca.n_components_} dimensions)")
                
            except ImportError:
                print("⚠️ sklearn not available, using manual alignment...")
                # Manual alignment using covariance matrix
                cov_matrix = np.cov(embedding_matrix.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                
                # Sort eigenvalues and eigenvectors
                idx = np.argsort(eigenvalues)[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                # Keep top 95% of variance
                total_variance = np.sum(eigenvalues)
                cumulative_variance = np.cumsum(eigenvalues) / total_variance
                n_components = np.argmax(cumulative_variance >= 0.95) + 1
                
                # Project embeddings
                aligned_embeddings = np.dot(embedding_matrix, eigenvectors[:, :n_components])
                
                # Update embeddings
                for i, node_id in enumerate(node_ids):
                    self.embeddings[node_id] = aligned_embeddings[i]
                
                print(f"✅ Embeddings aligned manually ({n_components} dimensions)")
            
            # Apply additional normalization
            self.normalize_embeddings()
            
        except Exception as e:
            print(f"❌ Error during alignment: {e}")
    
    def normalize_embeddings(self):
        """Normalize embeddings to unit length for better similarity calculations"""
        try:
            print("🔄 Normalizing embeddings...")
            
            for node_id in self.embeddings:
                embedding = self.embeddings[node_id]
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    self.embeddings[node_id] = embedding / norm
            
            # Verify normalization
            sample_norms = [np.linalg.norm(self.embeddings[node_id]) for node_id in list(self.embeddings.keys())[:5]]
            all_normalized = all(abs(norm - 1.0) < 0.01 for norm in sample_norms)
            
            if all_normalized:
                print("✅ Embeddings normalized successfully")
            else:
                print("⚠️ Some embeddings may not be properly normalized")
            
        except Exception as e:
            print(f"❌ Error normalizing embeddings: {e}")
    
    def enhance_embeddings_with_context(self):
        """Enhance embeddings using graph context information"""
        try:
            if not self.embeddings:
                print("⚠️ No embeddings to enhance")
                return
            
            print("🔄 Enhancing embeddings with graph context...")
            
            # Create enhanced embeddings using node degrees and centrality
            enhanced_embeddings = {}
            
            with self.driver.session() as session:
                # Calculate node centrality measures
                result = session.run("""
                    MATCH (n)
                    RETURN elementId(n) as node_id, 
                           COUNT { (n)--() } as degree,
                           COUNT { (n)-->() } as out_degree,
                           COUNT { ()--(n) } as in_degree
                """)
                
                centrality_data = {}
                for record in result:
                    node_id = record["node_id"]
                    centrality_data[node_id] = {
                        'degree': record["degree"],
                        'out_degree': record["out_degree"],
                        'in_degree': record["in_degree"]
                    }
                
                # Enhance embeddings with centrality information
                for node_identifier, embedding in self.embeddings.items():
                    # Extract elementId from node identifier
                    if "_" in node_identifier:
                        _, element_id = node_identifier.split("_", 1)
                        try:
                            element_id = int(element_id)
                        except ValueError:
                            element_id = element_id
                        
                        # Get centrality data
                        if element_id in centrality_data:
                            centrality = centrality_data[element_id]
                            
                            # Create enhancement vector based on centrality
                            # Ensure the enhancement vector has the same dimension as the embedding
                            enhancement_dim = len(embedding)
                            enhancement = np.zeros(enhancement_dim)
                            
                            # Fill the first 3 dimensions with centrality information
                            enhancement[0] = np.log(centrality['degree'] + 1)  # Log scale to reduce impact
                            if enhancement_dim > 1:
                                enhancement[1] = np.log(centrality['out_degree'] + 1)
                            if enhancement_dim > 2:
                                enhancement[2] = np.log(centrality['in_degree'] + 1)
                            
                            # Apply enhancement (small weight to preserve original structure)
                            enhanced_embedding = embedding + (enhancement * 0.1)
                            enhanced_embeddings[node_identifier] = enhanced_embedding
                        else:
                            enhanced_embeddings[node_identifier] = embedding
                    else:
                        enhanced_embeddings[node_identifier] = embedding
                
                # Update embeddings with enhanced versions
                self.embeddings = enhanced_embeddings
            
            print("✅ Embeddings enhanced with graph context")
            
        except Exception as e:
            print(f"❌ Error enhancing embeddings: {e}")
    
    def load_embeddings(self):
        """Load embeddings from the trained model"""
        try:
            if self.model:
                self.embeddings = {}
                for node in self.model.wv.index_to_key:
                    self.embeddings[node] = self.model.wv[node]
                print(f"✅ Loaded {len(self.embeddings)} embeddings")
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
    
    def get_detailed_node_info(self, node_identifier):
        """Get detailed information about a node from Neo4j"""
        try:
            # Extract the original node identifier from the format "Label_Name"
            if "_" in node_identifier:
                label, uid = node_identifier.split("_", 1)
                try:
                    uid = int(uid)  # Convert to int if it's a numeric uid
                except ValueError:
                    uid = uid  # Keep as string if it's a name
            else:
                label = node_identifier
                uid = None
            
            with self.driver.session() as session:
                if uid is not None:
                    # Try to query by elementId
                    result = session.run(f"""
                        MATCH (n:{label})
                        WHERE elementId(n) = $uid
                        RETURN n
                    """, uid=uid)
                else:
                    # Query by label only
                    result = session.run(f"""
                        MATCH (n:{label})
                        RETURN n
                        LIMIT 1
                    """)
                
                record = result.single()
                if record and record["n"]:
                    node = record["n"]
                    # Format node properties for display
                    properties = dict(node)
                    info_parts = [f"{label}: {properties.get('name', properties.get('title', 'Unnamed'))}"]
                    
                    # Add key properties
                    for key, value in properties.items():
                        if key not in ['name', 'title', 'id', 'element_id', 'uid'] and value:
                            info_parts.append(f"  {key}: {value}")
                    
                    return "\n".join(info_parts)
                
                # Fallback: try querying by name if uid didn't work
                if uid is not None:
                    result = session.run(f"""
                        MATCH (n:{label} {{name: $name}})
                        RETURN n
                    """, name=str(uid))
                    
                    record = result.single()
                    if record and record["n"]:
                        node = record["n"]
                        properties = dict(node)
                        info_parts = [f"{label}: {properties.get('name', properties.get('title', 'Unnamed'))}"]
                        
                        for key, value in properties.items():
                            if key not in ['name', 'title', 'id', 'element_id', 'uid'] and value:
                                info_parts.append(f"  {key}: {value}")
                        
                        return "\n".join(info_parts)
                
                return None
                
        except Exception as e:
            print(f"❌ Error getting detailed node info: {e}")
            return None
    
    def get_detailed_edge_info(self, source_node, target_node, edge_data):
        """Get detailed information about an edge from Neo4j"""
        try:
            rel_type = edge_data.get('type', 'unknown')
            rel_id = edge_data.get('element_id')
            
            with self.driver.session() as session:
                if rel_id:
                    # Query by relationship elementId
                    result = session.run("""
                        MATCH (s)-[r]->(t)
                        WHERE elementId(r) = $rel_id
                        RETURN s, r, t
                    """, rel_id=rel_id)
                    
                    record = result.single()
                    if record:
                        source = record["s"]
                        relationship = record["r"]
                        target = record["t"]
                        
                        # Format edge information
                        info_parts = [
                            f"Relationship: {rel_type}",
                            f"Source: {source.get('name', 'Unnamed')}",
                            f"Target: {target.get('name', 'Unnamed')}"
                        ]
                        
                        # Add relationship properties
                        properties = dict(relationship)
                        for key, value in properties.items():
                            if key not in ['element_id', 'type'] and value:
                                info_parts.append(f"  {key}: {value}")
                        
                        return "\n".join(info_parts)
                
                # Fallback: query by source and target
                if source_node and target_node:
                    # Extract elementIds from node identifiers
                    source_element_id = None
                    target_element_id = None
                    
                    for node_id in [source_node, target_node]:
                        if "_" in node_id:
                            _, element_id = node_id.split("_", 1)
                            try:
                                element_id = int(element_id)
                            except ValueError:
                                element_id = element_id
                            
                            if node_id == source_node:
                                source_element_id = element_id
                            else:
                                target_element_id = element_id
                    
                    if source_element_id and target_element_id:
                        result = session.run("""
                            MATCH (s)-[r]->(t)
                            WHERE elementId(s) = $source_id AND elementId(t) = $target_id AND type(r) = $rel_type
                            RETURN s, r, t
                        """, source_id=source_element_id, target_id=target_element_id, rel_type=rel_type)
                        
                        record = result.single()
                        if record:
                            source = record["s"]
                            relationship = record["r"]
                            target = record["t"]
                            
                            # Format edge information
                            info_parts = [
                                f"Relationship: {rel_type}",
                                f"Source: {source.get('name', 'Unnamed')}",
                                f"Target: {target.get('name', 'Unnamed')}"
                            ]
                            
                            # Add relationship properties
                            properties = dict(relationship)
                            for key, value in properties.items():
                                if key not in ['element_id', 'type'] and value:
                                    info_parts.append(f"  {key}: {value}")
                            
                            return "\n".join(info_parts)
                
                return None
                
        except Exception as e:
            print(f"❌ Error getting detailed edge info: {e}")
            return None

# Global instance
node2vec_service = Node2VecService()
