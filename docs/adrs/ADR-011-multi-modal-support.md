# ADR-011: Multi-Modal Support (Phase 2)

**Status**: Proposed (for Phase 2)  

**Date**: 2025-07-20  

**Deciders**: CodeForge AI Team  

## Context

Extend RAG to images/UI for Phase 2 web development. Phase 1 is text-only and needs vision capabilities for +20% accuracy in UI/web development tasks, processing screenshots, design mockups, and visual documentation.

## Problem Statement

Phase 1 text-only limitations need vision for enhanced UI/web development accuracy. Requirements include:

- Processing UI screenshots and design mockups

- Visual documentation analysis and understanding

- Integration with existing GraphRAG+ pipeline

- Support for image-based debugging and testing

- Seamless toggle for non-UI tasks to manage costs

## Decision

**OpenAI SDK for CLIP embeddings** in RAG/retrieval with vision integration for UI testing and workflow validation.

## Alternatives Considered

| Approach | Pros | Cons | Score |
|----------|------|------|-------|
| **OpenAI SDK + CLIP** | Easy integration, proven performance, good accuracy boost | API costs, dependency | **8.5** |
| Local CLIP models | No API costs, full control | Heavy setup, GPU requirements, maintenance | 7.5 |
| No multi-modal | Simpler, cost-effective | Misses significant value for UI/web tasks | 6.0 |
| Google Vision API | Good accuracy, integrated ecosystem | Higher costs, limited embedding control | 7.8 |

## Rationale

- **Easy integration and performance boost (8.5)**: Seamless addition to existing pipeline

- **Measured +20% accuracy**: Significant improvement for UI/web development tasks

- **Cost manageable**: Toggle for non-UI tasks controls spending

- **Future-proof**: Foundation for advanced visual reasoning in later phases

## Consequences

### Positive

- Significant accuracy improvement for UI/web development workflows

- Enhanced debugging capabilities with visual analysis

- Better integration testing through screenshot analysis

- Improved documentation understanding with visual content

### Negative

- Additional API costs for vision processing

- Increased complexity in RAG pipeline

- Potential latency increase for vision-enabled queries

### Neutral

- Toggle mechanism for non-UI tasks to control costs

- Gradual rollout to measure effectiveness

## Implementation Notes

### Multi-Modal RAG Architecture
```python
from openai import OpenAI
from typing import List, Dict, Optional, Union
import base64
import io
from PIL import Image
import numpy as np

class MultiModalRAG:
    def __init__(self, base_rag_system):
        self.base_rag = base_rag_system
        self.openai_client = OpenAI()
        self.vision_enabled = True
        self.image_cache = {}
        
        # Vision model configurations
        self.vision_models = {
            'clip': 'clip-vit-base-patch32',
            'gpt4v': 'gpt-4-vision-preview'
        }
        
        # Task types that benefit from vision
        self.vision_task_types = {
            'ui_analysis', 'web_testing', 'design_review', 
            'visual_debugging', 'screenshot_analysis'
        }
    
    async def retrieve_multimodal(self, query: str, images: List[str] = None, 
                                 task_type: str = 'general') -> List[Dict]:
        """Enhanced retrieval with optional image processing"""
        
        # Standard text-based retrieval
        text_results = await self.base_rag.retrieve(query)
        
        # Add vision processing if images provided and task benefits
        if images and task_type in self.vision_task_types and self.vision_enabled:
            vision_results = await self._process_images(query, images)
            
            # Fuse text and vision results
            fused_results = self._fuse_multimodal_results(
                text_results, vision_results, query
            )
            
            return fused_results
        
        return text_results
    
    async def _process_images(self, query: str, images: List[str]) -> List[Dict]:
        """Process images for enhanced context"""
        
        vision_results = []
        
        for image_path in images:
            try:
                # Load and preprocess image
                image_data = await self._load_image(image_path)
                
                # Generate embeddings using CLIP
                image_embedding = await self._generate_image_embedding(image_data)
                
                # Extract visual features and descriptions
                visual_description = await self._describe_image(image_data, query)
                
                # Create structured result
                vision_result = {
                    'type': 'image',
                    'path': image_path,
                    'embedding': image_embedding,
                    'description': visual_description,
                    'relevance_score': self._calculate_image_relevance(
                        visual_description, query
                    ),
                    'visual_features': await self._extract_visual_features(image_data)
                }
                
                vision_results.append(vision_result)
                
            except Exception as e:
                logger.error(f"Failed to process image {image_path}: {e}")
                continue
        
        return vision_results
    
    async def _generate_image_embedding(self, image_data: bytes) -> np.ndarray:
        """Generate CLIP embeddings for image"""
        
        # Convert to base64 for API
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        try:
            response = await self.openai_client.embeddings.create(
                model="clip-vit-base-patch32",
                input=f"data:image/jpeg;base64,{image_b64}"
            )
            
            return np.array(response.data[0].embedding)
            
        except Exception as e:
            logger.error(f"Failed to generate image embedding: {e}")
            return np.zeros(512)  # Fallback embedding
    
    async def _describe_image(self, image_data: bytes, context_query: str) -> str:
        """Generate detailed image description using GPT-4V"""
        
        image_b64 = base64.b64encode(image_data).decode('utf-8')
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Analyze this image in the context of: {context_query}\n\nProvide a detailed description focusing on:\n1. UI elements and layout (if applicable)\n2. Visual hierarchy and design patterns\n3. Technical details relevant to the query\n4. Any issues or improvements needed"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to describe image: {e}")
            return "Image description unavailable"
```

### Visual Feature Extraction
```python
class VisualFeatureExtractor:
    def __init__(self):
        self.ui_patterns = [
            'button', 'form', 'navigation', 'modal', 'dropdown',
            'card', 'table', 'chart', 'graph', 'menu'
        ]
        
        self.design_elements = [
            'color scheme', 'typography', 'spacing', 'alignment',
            'contrast', 'hierarchy', 'consistency'
        ]
    
    async def extract_ui_elements(self, image_description: str) -> Dict[str, List[str]]:
        """Extract UI elements from image description"""
        
        elements = {
            'components': [],
            'layout_patterns': [],
            'design_issues': [],
            'accessibility_concerns': []
        }
        
        description_lower = image_description.lower()
        
        # Identify UI components
        for pattern in self.ui_patterns:
            if pattern in description_lower:
                elements['components'].append(pattern)
        
        # Identify layout patterns
        layout_keywords = ['grid', 'flex', 'stack', 'sidebar', 'header', 'footer']
        for keyword in layout_keywords:
            if keyword in description_lower:
                elements['layout_patterns'].append(keyword)
        
        # Identify potential issues
        issue_keywords = ['overlap', 'cut off', 'unclear', 'hard to read', 'poor contrast']
        for keyword in issue_keywords:
            if keyword in description_lower:
                elements['design_issues'].append(keyword)
        
        return elements
    
    async def analyze_visual_hierarchy(self, image_description: str) -> Dict[str, float]:
        """Analyze visual hierarchy effectiveness"""
        
        hierarchy_scores = {
            'clarity': 0.5,
            'emphasis': 0.5,
            'flow': 0.5,
            'balance': 0.5
        }
        
        # Simple scoring based on description content
        # In production, this would use more sophisticated analysis
        
        positive_indicators = [
            'clear hierarchy', 'well organized', 'prominent', 'emphasized',
            'good flow', 'balanced', 'structured', 'logical'
        ]
        
        negative_indicators = [
            'cluttered', 'confusing', 'unclear', 'poor hierarchy',
            'unbalanced', 'chaotic', 'overwhelming'
        ]
        
        description_lower = image_description.lower()
        
        positive_count = sum(1 for indicator in positive_indicators 
                           if indicator in description_lower)
        negative_count = sum(1 for indicator in negative_indicators 
                           if indicator in description_lower)
        
        # Adjust scores based on indicators
        adjustment = (positive_count - negative_count) * 0.1
        
        for key in hierarchy_scores:
            hierarchy_scores[key] = max(0.0, min(1.0, 
                hierarchy_scores[key] + adjustment
            ))
        
        return hierarchy_scores
```

### Multi-Modal Fusion
```python
class MultiModalFusion:
    def __init__(self):
        self.fusion_weights = {
            'text_semantic': 0.4,
            'visual_content': 0.3,
            'text_visual_alignment': 0.2,
            'task_relevance': 0.1
        }
    
    def fuse_multimodal_results(self, text_results: List[Dict], 
                              vision_results: List[Dict], 
                              query: str) -> List[Dict]:
        """Intelligently fuse text and vision results"""
        
        fused_results = []
        
        # Add enhanced text results
        for text_result in text_results:
            enhanced_result = text_result.copy()
            enhanced_result['modality'] = 'text'
            enhanced_result['fusion_score'] = self._calculate_fusion_score(
                enhanced_result, None, query
            )
            fused_results.append(enhanced_result)
        
        # Add vision results
        for vision_result in vision_results:
            enhanced_result = vision_result.copy()
            enhanced_result['modality'] = 'vision'
            enhanced_result['fusion_score'] = self._calculate_fusion_score(
                None, enhanced_result, query
            )
            fused_results.append(enhanced_result)
        
        # Create cross-modal enriched results
        cross_modal_results = self._create_cross_modal_results(
            text_results, vision_results, query
        )
        fused_results.extend(cross_modal_results)
        
        # Sort by fusion score
        fused_results.sort(key=lambda x: x['fusion_score'], reverse=True)
        
        return fused_results[:20]  # Top 20 results
    
    def _calculate_fusion_score(self, text_result: Optional[Dict], 
                              vision_result: Optional[Dict], 
                              query: str) -> float:
        """Calculate relevance score for multimodal result"""
        
        score = 0.0
        
        if text_result:
            # Text semantic relevance
            score += text_result.get('relevance_score', 0.5) * self.fusion_weights['text_semantic']
            
            # Task relevance
            task_relevance = self._calculate_task_relevance(text_result, query)
            score += task_relevance * self.fusion_weights['task_relevance']
        
        if vision_result:
            # Visual content relevance
            score += vision_result.get('relevance_score', 0.5) * self.fusion_weights['visual_content']
        
        if text_result and vision_result:
            # Text-visual alignment
            alignment_score = self._calculate_text_visual_alignment(
                text_result, vision_result
            )
            score += alignment_score * self.fusion_weights['text_visual_alignment']
        
        return min(score, 1.0)
    
    def _create_cross_modal_results(self, text_results: List[Dict], 
                                  vision_results: List[Dict], 
                                  query: str) -> List[Dict]:
        """Create results that combine text and visual information"""
        
        cross_modal = []
        
        for text_result in text_results[:5]:  # Top 5 text results
            for vision_result in vision_results[:3]:  # Top 3 vision results
                
                # Calculate cross-modal relevance
                alignment_score = self._calculate_text_visual_alignment(
                    text_result, vision_result
                )
                
                if alignment_score > 0.6:  # Only combine if good alignment
                    combined_result = {
                        'type': 'cross_modal',
                        'text_component': text_result,
                        'visual_component': vision_result,
                        'combined_description': self._create_combined_description(
                            text_result, vision_result
                        ),
                        'fusion_score': alignment_score * 0.9,  # Slight penalty for complexity
                        'modality': 'multimodal'
                    }
                    
                    cross_modal.append(combined_result)
        
        return cross_modal
```

### Usage Patterns and Cost Management
```python
class VisionCostManager:
    def __init__(self):
        self.daily_budget = 25.0  # $25/day for vision features
        self.current_usage = 0.0
        self.cost_per_vision_call = 0.01  # Approximate cost
        
        # Task types that justify vision costs
        self.high_value_tasks = {
            'ui_testing', 'design_review', 'visual_debugging',
            'accessibility_audit', 'responsive_testing'
        }
    
    def should_use_vision(self, task_type: str, query: str, 
                         images_count: int) -> bool:
        """Determine if vision processing is cost-justified"""
        
        estimated_cost = images_count * self.cost_per_vision_call
        
        # Check budget
        if self.current_usage + estimated_cost > self.daily_budget:
            return False
        
        # High-value tasks always get vision
        if task_type in self.high_value_tasks:
            return True
        
        # Check for visual keywords in query
        visual_keywords = [
            'screenshot', 'image', 'visual', 'ui', 'interface',
            'design', 'layout', 'appearance', 'color', 'style'
        ]
        
        if any(keyword in query.lower() for keyword in visual_keywords):
            return True
        
        return False
    
    def record_usage(self, cost: float):
        """Record vision API usage"""
        self.current_usage += cost
        
        # Log usage for monitoring
        logger.info(f"Vision API usage: +${cost:.3f}, total: ${self.current_usage:.3f}")
```

## Integration with Existing Systems

### GraphRAG+ Enhancement
```python
class VisionEnhancedGraphRAG:
    def __init__(self, base_graphrag, vision_rag):
        self.base_graphrag = base_graphrag
        self.vision_rag = vision_rag
    
    async def enhanced_retrieve(self, query: str, images: List[str] = None,
                              task_type: str = 'general') -> List[Dict]:
        """Enhanced retrieval with vision integration"""
        
        # Get base results
        base_results = await self.base_graphrag.retrieve(query)
        
        # Add vision enhancement if applicable
        if images and self.vision_rag.should_use_vision(task_type, query, len(images)):
            vision_enhanced = await self.vision_rag.retrieve_multimodal(
                query, images, task_type
            )
            
            # Merge results with appropriate weighting
            return self._merge_enhanced_results(base_results, vision_enhanced)
        
        return base_results
    
    def _merge_enhanced_results(self, base_results: List[Dict], 
                              vision_results: List[Dict]) -> List[Dict]:
        """Merge base and vision-enhanced results"""
        
        # Weight vision results higher for visual tasks
        for result in vision_results:
            if result.get('modality') in ['vision', 'multimodal']:
                result['relevance_score'] *= 1.2  # 20% boost for visual content
        
        # Combine and sort
        all_results = base_results + vision_results
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return all_results[:15]  # Top 15 results
```

## Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Accuracy Improvement | +20% | UI/web task benchmarks |
| Vision Processing Latency | <2s | Average per image |
| Daily Cost Impact | <$25 | Vision API usage |
| Task Coverage | 80% | Vision-enabled vs total UI tasks |
| User Satisfaction | +25% | UI/web development workflows |

## Phase 2 Rollout Strategy

### Gradual Enablement
1. **Week 1**: Enable for design review tasks only
2. **Week 2**: Add UI testing and debugging workflows  
3. **Week 3**: Expand to accessibility audits
4. **Week 4**: Full enablement with cost monitoring

### A/B Testing

- 50% of UI/web tasks use vision enhancement

- Track accuracy, cost, and user satisfaction metrics

- Adjust based on performance data

## Related Decisions

- ADR-006: SOTA GraphRAG Implementation

- ADR-008: Multi-Model Routing

- ADR-003: Tool Integration Protocol

## Monitoring

- Vision API usage and cost tracking

- Accuracy improvements on visual tasks

- User adoption and satisfaction metrics

- Processing latency and performance

- Cost per accuracy improvement ratio
