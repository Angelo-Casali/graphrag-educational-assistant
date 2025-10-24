#!/usr/bin/env python3
"""
Educational Context Builder for GraphRAG
Transforms raw graph retrieval results into structured educational context
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

logger = logging.getLogger(__name__)

class ConfidenceLevel(Enum):
    """Confidence levels for educational recommendations"""
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH" 
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"

@dataclass
class MethodologyRecommendation:
    """A single methodology recommendation with context"""
    name: str
    category: str
    relevance_score: float
    evidence_type: str  # 'direct_relationship', 'semantic_similarity', 'domain_knowledge'
    implementation_guidance: str
    classroom_applications: List[str]
    special_considerations: List[str]
    confidence: ConfidenceLevel

@dataclass
class StudentProfile:
    """Student profile extracted from query and context"""
    primary_needs: List[str]
    secondary_needs: List[str]
    educational_context: str  # 'special_needs', 'assessment', 'general'
    grade_level: Optional[str]
    subject_area: Optional[str]

@dataclass
class EducationalContext:
    """Complete educational context for response generation"""
    student_profile: StudentProfile
    primary_methodologies: List[MethodologyRecommendation]
    supporting_methodologies: List[MethodologyRecommendation]
    evidence_summary: str
    implementation_priority: List[str]
    confidence_assessment: ConfidenceLevel
    fallback_strategies: List[str]
    metadata: Dict[str, Any]

class PedagogicalKnowledgeBase:
    """Neuroscience domain knowledge and teaching applications"""
    
    def __init__(self):
        self.methodology_categories = {
            'IntrinsicMotivation': {
                'category': 'Motivational Systems',
                'best_for': ['engagement', 'self_directed_learning', 'persistence'],
                'implementation': 'Foster autonomy, competence, and relatedness in learning',
                'applications': [
                    'Offer choice in learning activities',
                    'Connect content to student interests and real-world applications',
                    'Set personalized, meaningful learning goals'
                ],
                'special_needs_adaptations': [
                    'Celebrate progress and effort over outcomes',
                    'Use authentic, relevant assessment methods',
                    'Build on student strengths and interests'
                ]
            },
            'ExtrinsicMotivation': {
                'category': 'Motivational Systems',
                'best_for': ['initial_engagement', 'behavior_management', 'skill_acquisition'],
                'implementation': 'Use external rewards strategically and transition to intrinsic motivation',
                'applications': [
                    'Provide immediate feedback and recognition',
                    'Use gamification elements for engagement',
                    'Gradually fade external rewards'
                ],
                'special_needs_adaptations': [
                    'Pair rewards with intrinsic satisfaction',
                    'Focus on mastery rather than performance',
                    'Transition to self-directed goal setting'
                ]
            },
            'GrowthMindset': {
                'category': 'Mindset & Beliefs',
                'best_for': ['resilience', 'learning_from_failure', 'challenge_seeking'],
                'implementation': 'Teach brain plasticity and the power of "yet"',
                'applications': [
                    'Reframe mistakes as learning opportunities',
                    'Praise effort, strategies, and progress',
                    'Share stories of growth and improvement'
                ],
                'special_needs_adaptations': [
                    'Model growth mindset language consistently',
                    'Provide specific feedback on learning strategies',
                    'Celebrate small wins and incremental progress'
                ]
            },
            'FixedMindset': {
                'category': 'Mindset & Beliefs',
                'best_for': ['awareness', 'intervention', 'mindset_shift'],
                'implementation': 'Recognize and challenge fixed mindset beliefs',
                'applications': [
                    'Identify fixed mindset triggers in students',
                    'Teach about neuroplasticity explicitly',
                    'Provide counter-examples of ability development'
                ],
                'special_needs_adaptations': [
                    'Create safe environment for risk-taking',
                    'Normalize struggle as part of learning',
                    'Use growth-oriented language'
                ]
            },
            'Metacognition': {
                'category': 'Self-Regulation & Awareness',
                'best_for': ['planning', 'monitoring', 'reflection', 'strategy_use'],
                'implementation': 'Explicitly teach thinking about thinking',
                'applications': [
                    'Use think-aloud protocols during problem-solving',
                    'Implement reflection journals and self-assessment',
                    'Teach specific learning strategies explicitly'
                ],
                'special_needs_adaptations': [
                    'Provide metacognitive question prompts',
                    'Use checklists for self-monitoring',
                    'Model metacognitive thinking regularly'
                ]
            },
            'PositiveStressEustress': {
                'category': 'Stress & Arousal',
                'best_for': ['optimal_challenge', 'engagement', 'performance'],
                'implementation': 'Create optimal challenge level (Goldilocks zone)',
                'applications': [
                    'Set appropriately challenging tasks',
                    'Provide support structures for complex tasks',
                    'Frame challenges as growth opportunities'
                ],
                'special_needs_adaptations': [
                    'Monitor stress levels during activities',
                    'Teach stress management techniques',
                    'Balance challenge with adequate support'
                ]
            },
            'NegativeStressDistress': {
                'category': 'Stress & Arousal',
                'best_for': ['stress_reduction', 'anxiety_management', 'wellbeing'],
                'implementation': 'Reduce excessive stress and create safe learning environment',
                'applications': [
                    'Implement calming routines before assessments',
                    'Teach breathing and relaxation techniques',
                    'Reduce time pressure when possible'
                ],
                'special_needs_adaptations': [
                    'Provide predictable routines and structures',
                    'Offer breaks and movement opportunities',
                    'Create emotionally safe classroom climate'
                ]
            },
            'PositiveEmotions': {
                'category': 'Emotional Systems',
                'best_for': ['engagement', 'memory', 'creativity', 'motivation'],
                'implementation': 'Foster positive emotional climate in classroom',
                'applications': [
                    'Build positive teacher-student relationships',
                    'Celebrate successes and learning moments',
                    'Use humor and joy in teaching'
                ],
                'special_needs_adaptations': [
                    'Create emotionally safe environment',
                    'Acknowledge and validate student emotions',
                    'Use positive emotion to enhance memory'
                ]
            },
            'Attention': {
                'category': 'Cognitive Processes',
                'best_for': ['focus', 'information_processing', 'learning_efficiency'],
                'implementation': 'Design lessons to capture and maintain attention',
                'applications': [
                    'Use varied stimuli and teaching methods',
                    'Minimize distractions in learning environment',
                    'Chunk information into manageable segments'
                ],
                'special_needs_adaptations': [
                    'Provide movement breaks regularly',
                    'Use visual and auditory cues strategically',
                    'Teach attention self-monitoring strategies'
                ]
            },
            'WorkingMemory': {
                'category': 'Memory Systems',
                'best_for': ['information_retention', 'problem_solving', 'comprehension'],
                'implementation': 'Reduce cognitive load and support working memory',
                'applications': [
                    'Chunk information into smaller units',
                    'Use visual aids and graphic organizers',
                    'Provide written instructions alongside verbal'
                ],
                'special_needs_adaptations': [
                    'Limit amount of new information presented',
                    'Allow use of external memory aids',
                    'Provide repetition and rehearsal opportunities'
                ]
            },
            'ExecutiveFunctions': {
                'category': 'Cognitive Control',
                'best_for': ['planning', 'organization', 'self_regulation', 'goal_achievement'],
                'implementation': 'Scaffold executive function development',
                'applications': [
                    'Teach planning and organizational strategies explicitly',
                    'Use checklists and visual schedules',
                    'Break complex tasks into steps'
                ],
                'special_needs_adaptations': [
                    'Provide external organizational tools',
                    'Model executive function strategies',
                    'Gradually transfer responsibility to students'
                ]
            },
            'CriticalThinking': {
                'category': 'Higher-Order Thinking',
                'best_for': ['analysis', 'evaluation', 'problem_solving'],
                'implementation': 'Teach critical thinking skills explicitly',
                'applications': [
                    'Use Socratic questioning methods',
                    'Teach argument analysis and evaluation',
                    'Provide opportunities for debate and discussion'
                ],
                'special_needs_adaptations': [
                    'Scaffold critical thinking with graphic organizers',
                    'Model thinking processes explicitly',
                    'Start with concrete examples'
                ]
            }
        }
        
        self.special_needs_mapping = {
            'Motivation': ['engagement_strategies', 'relevance_connection', 'choice_provision'],
            'Attention': ['focus_strategies', 'minimize_distractions', 'varied_stimuli'],
            'WorkingMemory': ['chunking', 'repetition', 'visual_aids', 'reduced_cognitive_load'],
            'ExecutiveFunctions': ['scaffolding', 'explicit_instruction', 'organizational_tools'],
            'PositiveEmotions': ['safe_environment', 'positive_relationships', 'success_experiences'],
            'NegativeEmotions': ['emotion_regulation', 'coping_strategies', 'emotional_support'],
            'PositiveStressEustress': ['optimal_challenge', 'growth_zone', 'support_structures'],
            'NegativeStressDistress': ['stress_reduction', 'anxiety_management', 'calm_environment'],
            'GrowthMindset': ['praise_effort', 'normalize_struggle', 'celebrate_progress'],
            'FixedMindset': ['challenge_beliefs', 'teach_neuroplasticity', 'reframe_failure'],
            'Metacognition': ['self_monitoring', 'reflection_tools', 'strategy_instruction'],
            'Memory': ['encoding_strategies', 'retrieval_practice', 'spaced_repetition'],
            'Creativity': ['divergent_thinking', 'open_ended_tasks', 'brainstorming'],
            'CriticalThinking': ['questioning_techniques', 'analysis_frameworks', 'evaluation_criteria']
        }
        
        self.fallback_strategies = {
            'no_results': [
                'Evidence-based teaching practices grounded in learning sciences',
                'Brain-friendly learning environment design',
                'Multi-modal instruction techniques',
                'Growth mindset cultivation strategies'
            ],
            'low_confidence': [
                'Consult neuroscience education specialists or educational psychologists',
                'Review current research on learning sciences and cognitive neuroscience',
                'Implement evidence-based strategies gradually with monitoring',
                'Seek peer teacher collaboration and professional development'
            ]
        }

class MethodologyRanker:
    """Ranks and prioritizes educational methodologies"""
    
    def __init__(self, knowledge_base: PedagogicalKnowledgeBase):
        self.kb = knowledge_base
    
    def rank_methodologies(self, nodes: List[Dict], query_metadata: Dict) -> List[MethodologyRecommendation]:
        """Rank methodologies based on relevance and evidence"""
        recommendations = []
        
        for node in nodes:
            if self._is_methodology(node):
                recommendation = self._create_recommendation(node, query_metadata)
                if recommendation:
                    recommendations.append(recommendation)
        
        # Sort by relevance score (descending)
        recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return recommendations
    
    def _is_methodology(self, node: Dict) -> bool:
        """Check if node represents a pedagogical methodology"""
        labels = node.get('labels', [])
        return any(label in ['PedagogicalMethodology', 'TeachingApproach', 'LearningStrategy'] 
                  for label in labels)
    
    def _create_recommendation(self, node: Dict, query_metadata: Dict) -> Optional[MethodologyRecommendation]:
        """Create a methodology recommendation from a node"""
        name = node.get('name', '')
        if not name:
            return None
        
        # Get knowledge base info
        kb_info = self.kb.methodology_categories.get(name, {})
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(node, query_metadata)
        
        # Determine evidence type
        evidence_type = self._determine_evidence_type(node)
        
        # Get implementation guidance
        implementation = kb_info.get('implementation', f'Apply {name} methodology with appropriate adaptations')
        
        # Get classroom applications
        applications = kb_info.get('applications', [f'Implement {name} in classroom context'])
        
        # Get special considerations
        special_considerations = kb_info.get('special_needs_adaptations', ['Adapt based on individual student needs'])
        
        # Determine confidence
        confidence = self._calculate_confidence(relevance_score, evidence_type, kb_info)
        
        return MethodologyRecommendation(
            name=name,
            category=kb_info.get('category', 'Educational Methodology'),
            relevance_score=relevance_score,
            evidence_type=evidence_type,
            implementation_guidance=implementation,
            classroom_applications=applications,
            special_considerations=special_considerations,
            confidence=confidence
        )
    
    def _calculate_relevance_score(self, node: Dict, query_metadata: Dict) -> float:
        """Calculate relevance score for a methodology"""
        base_score = 0.5
        
        # Boost for semantic similarity
        if node.get('source') == 'semantic':
            semantic_score = node.get('semantic_score', 0.5)
            base_score += semantic_score * 0.3
        
        # Boost for direct graph relationships
        if node.get('source') == 'graph' or node.get('rel_type'):
            base_score += 0.4
        
        # Boost for Node2Vec vector similarity
        if 'vector_similarity' in node:
            vector_score = node.get('vector_similarity', 0.0)
            base_score += vector_score * 0.2
        
        # Context-specific boosts
        if query_metadata.get('educational_context') == 'special_needs':
            if any(adaptation in str(node).lower() 
                  for adaptation in ['inclusive', 'adaptive', 'support']):
                base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _determine_evidence_type(self, node: Dict) -> str:
        """Determine the type of evidence supporting this recommendation"""
        if node.get('rel_type'):
            return 'direct_relationship'
        elif node.get('source') == 'semantic':
            return 'semantic_similarity'
        elif node.get('vector_similarity'):
            return 'vector_similarity'
        else:
            return 'domain_knowledge'
    
    def _calculate_confidence(self, relevance_score: float, evidence_type: str, kb_info: Dict) -> ConfidenceLevel:
        """Calculate confidence level for recommendation"""
        if relevance_score >= 0.8 and evidence_type == 'direct_relationship':
            return ConfidenceLevel.VERY_HIGH
        elif relevance_score >= 0.7:
            return ConfidenceLevel.HIGH
        elif relevance_score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif relevance_score >= 0.3:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

class EvidenceSynthesizer:
    """Synthesizes evidence from graph relationships and semantic similarities"""
    
    def synthesize_evidence(self, triples: List[Dict], nodes: List[Dict]) -> str:
        """Create evidence summary from relationships and nodes"""
        evidence_parts = []
        
        # Analyze direct relationships
        relationship_evidence = self._analyze_relationships(triples)
        if relationship_evidence:
            evidence_parts.append(f"Direct pedagogical evidence: {relationship_evidence}")
        
        # Analyze semantic similarities
        semantic_evidence = self._analyze_semantic_nodes(nodes)
        if semantic_evidence:
            evidence_parts.append(f"Semantic analysis: {semantic_evidence}")
        
        # Analyze vector similarities
        vector_evidence = self._analyze_vector_similarities(nodes)
        if vector_evidence:
            evidence_parts.append(f"Conceptual similarity: {vector_evidence}")
        
        if not evidence_parts:
            return "Recommendations based on general pedagogical principles and domain expertise."
        
        return " | ".join(evidence_parts)
    
    def _analyze_relationships(self, triples: List[Dict]) -> str:
        """Analyze direct graph relationships"""
        if not triples:
            return ""
        
        suggests_count = sum(1 for t in triples if 'SUGGESTS' in t.get('relationship', ''))
        applies_count = sum(1 for t in triples if 'APPLIES_TO' in t.get('relationship', ''))
        
        parts = []
        if suggests_count:
            parts.append(f"{suggests_count} direct methodology suggestions")
        if applies_count:
            parts.append(f"{applies_count} application contexts")
        
        return ", ".join(parts)
    
    def _analyze_semantic_nodes(self, nodes: List[Dict]) -> str:
        """Analyze semantic similarity nodes"""
        semantic_nodes = [n for n in nodes if n.get('source') == 'semantic']
        if not semantic_nodes:
            return ""
        
        return f"Found {len(semantic_nodes)} semantically related educational concepts"
    
    def _analyze_vector_similarities(self, nodes: List[Dict]) -> str:
        """Analyze vector similarity evidence"""
        vector_nodes = [n for n in nodes if 'vector_similarity' in n]
        if not vector_nodes:
            return ""
        
        avg_similarity = sum(n.get('vector_similarity', 0) for n in vector_nodes) / len(vector_nodes)
        return f"Average conceptual similarity of {avg_similarity:.2f} across {len(vector_nodes)} related concepts"

class EducationalContextBuilder:
    """Main context builder that orchestrates the transformation"""
    
    def __init__(self):
        self.knowledge_base = PedagogicalKnowledgeBase()
        self.methodology_ranker = MethodologyRanker(self.knowledge_base)
        self.evidence_synthesizer = EvidenceSynthesizer()
    
    async def build_context(
        self, 
        retrieval_result: Dict, 
        original_query: str, 
        query_metadata: Dict
    ) -> EducationalContext:
        """Build comprehensive educational context from retrieval results"""
        
        try:
            logger.info(f"Building context for query: {original_query[:50]}...")
            
            # Extract components
            nodes = retrieval_result.get('nodes', [])
            triples = retrieval_result.get('triples', [])
            metadata = retrieval_result.get('metadata', {})
            
            # Build student profile
            student_profile = self._build_student_profile(original_query, query_metadata, nodes)
            
            # Rank methodologies
            all_recommendations = self.methodology_ranker.rank_methodologies(nodes, query_metadata)
            
            # Split into primary and supporting
            primary_methodologies = all_recommendations[:3]  # Top 3
            supporting_methodologies = all_recommendations[3:6]  # Next 3
            
            # Synthesize evidence
            evidence_summary = self.evidence_synthesizer.synthesize_evidence(triples, nodes)
            
            # Determine implementation priority
            implementation_priority = self._determine_implementation_priority(
                primary_methodologies, student_profile
            )
            
            # Calculate overall confidence
            confidence_assessment = self._calculate_overall_confidence(
                all_recommendations, len(triples), metadata
            )
            
            # Get fallback strategies if needed
            fallback_strategies = self._get_fallback_strategies(
                confidence_assessment, student_profile
            )
            
            context = EducationalContext(
                student_profile=student_profile,
                primary_methodologies=primary_methodologies,
                supporting_methodologies=supporting_methodologies,
                evidence_summary=evidence_summary,
                implementation_priority=implementation_priority,
                confidence_assessment=confidence_assessment,
                fallback_strategies=fallback_strategies,
                metadata={
                    'total_nodes': len(nodes),
                    'total_triples': len(triples),
                    'semantic_nodes': metadata.get('semantic_count', 0),
                    'graph_nodes': metadata.get('graph_count', 0),
                    'original_query': original_query,
                    'query_type': query_metadata.get('educational_context', 'general')
                }
            )
            
            logger.info(f"Context built successfully with {len(primary_methodologies)} primary recommendations")
            return context
            
        except Exception as e:
            logger.error(f"Error building educational context: {e}")
            return self._create_fallback_context(original_query, query_metadata)
    
    def _build_student_profile(self, query: str, metadata: Dict, nodes: List[Dict]) -> StudentProfile:
        """Build student profile from query and context"""
        
        # Extract needs from query and nodes
        primary_needs = []
        secondary_needs = []
        
        # Map Italian terms to educational needs
        query_lower = query.lower()
        for term, needs in self.knowledge_base.special_needs_mapping.items():
            if any(need_keyword in query_lower for need_keyword in [
                'ipovedenti', 'ciechi', 'blind',
                'sord', 'deaf', 'uditiv',
                'disabilità fisica', 'physical',
                'cognitive', 'cognitiv',
                'adhd', 'attenzione', 'attention',
                'autis', 'spettro',
                'motivazione', 'motivation'
            ]):
                if term.lower() in query_lower or any(alt in query_lower for alt in needs):
                    primary_needs.extend(needs[:1])  # Primary need
                    secondary_needs.extend(needs[1:])  # Secondary needs
        
        # Extract from node names
        for node in nodes:
            node_name = node.get('name', '').lower()
            for term, needs in self.knowledge_base.special_needs_mapping.items():
                if term.lower() in node_name:
                    if needs[0] not in primary_needs:
                        primary_needs.append(needs[0])
        
        # Determine educational context
        educational_context = metadata.get('educational_context', 'general')
        if any(term in query_lower for term in ['special', 'disabilità', 'difficoltà', 'bisogni']):
            educational_context = 'special_needs'
        elif any(term in query_lower for term in ['verific', 'valut', 'test', 'esam']):
            educational_context = 'assessment'
        
        return StudentProfile(
            primary_needs=list(set(primary_needs)),
            secondary_needs=list(set(secondary_needs)),
            educational_context=educational_context,
            grade_level=None,  # Could be extracted if available
            subject_area=None   # Could be extracted if available
        )
    
    def _determine_implementation_priority(
        self, 
        methodologies: List[MethodologyRecommendation], 
        student_profile: StudentProfile
    ) -> List[str]:
        """Determine implementation priority order"""
        
        if not methodologies:
            return ["Consult with educational specialists for personalized recommendations"]
        
        priority_order = []
        
        # High-confidence methodologies first
        high_confidence = [m for m in methodologies if m.confidence in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH]]
        if high_confidence:
            priority_order.append(f"Start with {high_confidence[0].name} (high confidence)")
        
        # Special needs considerations
        if student_profile.educational_context == 'special_needs':
            priority_order.append("Ensure accessibility accommodations are in place")
            priority_order.append("Begin with small-group implementation")
        
        # General implementation advice
        priority_order.extend([
            "Pilot with a subset of students first",
            "Gather feedback and adjust based on student response",
            "Gradually expand implementation across all relevant contexts"
        ])
        
        return priority_order
    
    def _calculate_overall_confidence(
        self, 
        recommendations: List[MethodologyRecommendation], 
        triple_count: int, 
        metadata: Dict
    ) -> ConfidenceLevel:
        """Calculate overall confidence in recommendations"""
        
        if not recommendations:
            return ConfidenceLevel.VERY_LOW
        
        # Average confidence of recommendations
        confidence_scores = {
            ConfidenceLevel.VERY_HIGH: 5,
            ConfidenceLevel.HIGH: 4,
            ConfidenceLevel.MEDIUM: 3,
            ConfidenceLevel.LOW: 2,
            ConfidenceLevel.VERY_LOW: 1
        }
        
        avg_confidence = sum(confidence_scores[r.confidence] for r in recommendations) / len(recommendations)
        
        # Boost for relationship evidence
        if triple_count > 0:
            avg_confidence += 0.5
        
        # Boost for semantic/vector evidence
        semantic_count = metadata.get('semantic_count', 0)
        if semantic_count > 5:
            avg_confidence += 0.3
        
        # Map back to confidence levels
        if avg_confidence >= 4.5:
            return ConfidenceLevel.VERY_HIGH
        elif avg_confidence >= 3.5:
            return ConfidenceLevel.HIGH
        elif avg_confidence >= 2.5:
            return ConfidenceLevel.MEDIUM
        elif avg_confidence >= 1.5:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _get_fallback_strategies(
        self, 
        confidence: ConfidenceLevel, 
        student_profile: StudentProfile
    ) -> List[str]:
        """Get fallback strategies based on confidence and context"""
        
        fallbacks = []
        
        if confidence in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]:
            fallbacks.extend(self.knowledge_base.fallback_strategies['low_confidence'])
        
        if not student_profile.primary_needs:
            fallbacks.extend(self.knowledge_base.fallback_strategies['no_results'])
        
        # Context-specific fallbacks
        if student_profile.educational_context == 'special_needs':
            fallbacks.extend([
                "Consult Individualized Education Program (IEP) if available",
                "Consider assistive technology options",
                "Collaborate with special education support team"
            ])
        
        return list(set(fallbacks))  # Remove duplicates
    
    def _create_fallback_context(self, query: str, metadata: Dict) -> EducationalContext:
        """Create a fallback context when primary building fails"""
        
        fallback_methodologies = [
            MethodologyRecommendation(
                name="Universal Design for Learning",
                category="Inclusive Pedagogy",
                relevance_score=0.6,
                evidence_type="domain_knowledge",
                implementation_guidance="Apply UDL principles to make learning accessible to all students",
                classroom_applications=["Multiple means of representation", "Multiple means of engagement", "Multiple means of expression"],
                special_considerations=["Flexible content delivery", "Choice in learning activities", "Varied assessment methods"],
                confidence=ConfidenceLevel.MEDIUM
            )
        ]
        
        return EducationalContext(
            student_profile=StudentProfile(
                primary_needs=["general_support"],
                secondary_needs=[],
                educational_context=metadata.get('educational_context', 'general'),
                grade_level=None,
                subject_area=None
            ),
            primary_methodologies=fallback_methodologies,
            supporting_methodologies=[],
            evidence_summary="Fallback recommendations based on general pedagogical principles",
            implementation_priority=["Apply universal design principles", "Consult with educational specialists"],
            confidence_assessment=ConfidenceLevel.LOW,
            fallback_strategies=self.knowledge_base.fallback_strategies['no_results'],
            metadata={'fallback': True, 'original_query': query}
        )

# Example usage
if __name__ == "__main__":
    import asyncio
    import logging
    import json
    from dataclasses import asdict
    
    async def test_context_builder():
        builder = EducationalContextBuilder()
        
        # Mock retrieval result
        mock_result = {
            'nodes': [
                {
                    'name': 'Cooperative Learning',
                    'labels': ['PedagogicalMethodology'],
                    'source': 'graph',
                    'rel_type': 'SUGGESTS'
                },
                {
                    'name': 'Flipped Classroom', 
                    'labels': ['PedagogicalMethodology'],
                    'source': 'semantic',
                    'semantic_score': 0.8
                }
            ],
            'triples': [
                {'relationship': 'SUGGESTS', 'source': 'Blind', 'target': 'Cooperative Learning'}
            ],
            'metadata': {'semantic_count': 1, 'graph_count': 1}
        }
        
        context = await builder.build_context(
            mock_result,
            "Il mio studente ha l'ADHD, cosa posso fare?",
            {'educational_context': 'special_needs'}
        )
        
        # Pretty-print the full context as JSON (ideal output shape)
        print("\n=== EDUCATIONAL CONTEXT (Full JSON) ===")
        print(json.dumps(asdict(context), indent=2, default=str))
        
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_context_builder())
