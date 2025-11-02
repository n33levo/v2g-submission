"""
Part 2: AI/ML and Agentic Workflow Design (Conceptual Design)
LLM-Based Variant-to-Gene (V2G) Relationship Agent

This module focuses on conceptual clarity, matching the technical assessment brief:
*Tool orchestration → evidence scoring → self-verification → optional LLM synthesis.*
The complementary architecture diagram lives in `architecture.md`.

Author: Neel Sarkar
Date: November 2025
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import pandas as pd

# ============================================================================
# CONCEPTUAL FRAMEWORK: Agent Architecture Design
# ============================================================================

@dataclass
class APIQuery:
    """Standard structure for all biological API queries."""
    endpoint: str
    parameters: Dict[str, Any]
    expected_fields: List[str]
    confidence_weight: float  # How much to trust this source


@dataclass
class Evidence:
    """Standardized evidence format from any biological data source."""
    source_api: str
    evidence_type: str
    data: Dict[str, Any]
    confidence_score: float  # 0.0 to 1.0
    query_metadata: Dict[str, Any]


class BiologicalAPITool(ABC):
    """
    Abstract tool interface - key design principle for extensibility.
    
    Why this design?
    - Each biological database has different API structure
    - New data sources can be added without changing core agent logic
    - Standardizes confidence scoring across diverse evidence types
    """
    
    @abstractmethod
    def construct_query(self, variant_id: str, gene_symbol: str) -> APIQuery:
        """
        Build the API-specific query structure for a variant–gene request.

        WHY: GTEx, gnomAD, and UniProt expose different endpoints and parameters.
        The agent relies on each tool to encode that knowledge so orchestration
        logic stays generic.
        """
        raise NotImplementedError

    @abstractmethod
    def parse_response(self, response: Dict) -> Evidence:
        """
        Normalize a raw API response into the canonical `Evidence` object.

        WHY: Downstream scoring expects consistent keys (source, content,
        confidence), so each tool takes responsibility for transforming its
        native format.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_confidence(self, raw_data: Dict) -> float:
        """
        Quantify the trustworthiness of the evidence returned by this tool.

        WHY: Different databases signal confidence differently (p-values,
        constraint scores, manual curation). Each tool documents and computes
        its domain-specific scoring logic.
        """
        raise NotImplementedError


# ============================================================================
# CONCRETE TOOL IMPLEMENTATIONS (Pseudocode Focus)
# ============================================================================

class GTExAPI(BiologicalAPITool):
    """
    GTEx Portal API for eQTL (expression quantitative trait loci) data.
    
    Purpose: Determine if variant affects gene expression in relevant tissues.
    API Documentation: https://gtexportal.org/api/v2/
    """
    
    def construct_query(self, variant_id: str, gene_symbol: str) -> APIQuery:
        """
        PSEUDOCODE for GTEx API interaction:
        
        Real implementation would query:
        GET https://gtexportal.org/api/v2/association/dyneqtl
        Parameters: geneId={gene_symbol}, snpId={variant_id}
        """
        return APIQuery(
            endpoint="https://gtexportal.org/api/v2/association/dyneqtl",
            parameters={
                "geneId": gene_symbol,
                "snpId": variant_id,
                "tissueName": "all",  # Agent would select relevant tissues
                "format": "json"
            },
            expected_fields=["pValue", "beta", "tissueName", "geneSymbol"],
            confidence_weight=0.8  # High trust in GTEx data quality
        )
    
    def parse_response(self, response: Dict) -> Evidence:
        """
        PSEUDOCODE for response parsing:
        
        Real implementation would:
        1. Extract significant associations (p < 0.05)
        2. Focus on relevant tissues (e.g., heart for cardiac genes)
        3. Calculate effect size magnitude
        
        EDUCATIONAL NOTE: This demonstrates how agent adapts evidence
        quality based on gene-context matching.
        """
        context = response.get("query", {})
        gene_symbol = context.get("gene_symbol", "").upper()
        
        # Agent's knowledge base for context-specific evidence
        cardiac_genes = {"MYBPC3", "MYH7", "TNNT2", "TPM1"}
        metabolic_genes = {"LRMDA", "MGMT"}
        
        # ADAPTIVE EVIDENCE GENERATION based on biological context
        if gene_symbol == "MYBPC3":  # Known HCM gene - moderate evidence
            associations = [
                {"tissue": "Heart-LeftVentricle", "pValue": 2e-5, "beta": -0.25},
                {"tissue": "Heart-AtrialAppendage", "pValue": 0.003, "beta": -0.18}
            ]
        elif gene_symbol in cardiac_genes:  # Other cardiac genes - variable evidence
            associations = [
                {"tissue": "Heart-LeftVentricle", "pValue": 1e-4, "beta": -0.19},
                {"tissue": "WholeBlood", "pValue": 0.02, "beta": 0.11}
            ]
        elif gene_symbol in metabolic_genes:  # Metabolic genes - tissue specific
            associations = [
                {"tissue": "Liver", "pValue": 8e-4, "beta": 0.16},
                {"tissue": "Adipose-Subcutaneous", "pValue": 0.015, "beta": 0.12}
            ]
        else:  # Novel/unknown genes - weak evidence
            associations = [
                {"tissue": "WholeBlood", "pValue": 0.08, "beta": 0.05}  # Non-significant
            ]
        
        mock_eqtl_data = {
            "significant_associations": associations,
            "tissue_count": len(associations),
            "max_effect_size": max(abs(a["beta"]) for a in associations)
        }
        
        return Evidence(
            source_api="GTEx",
            evidence_type="gene_expression_qtl",
            data=mock_eqtl_data,
            confidence_score=self.calculate_confidence(mock_eqtl_data),
            query_metadata={"version": "v8", "query_date": "2025-11-02"}
        )
    
    def calculate_confidence(self, raw_data: Dict) -> float:
        """
        GTEx-specific confidence scoring logic.
        
        EDUCATIONAL: Shows how real confidence scoring works:
        - Statistical significance (p-values)
        - Effect size magnitude  
        - Tissue relevance
        - Reproducibility across tissues
        """
        associations = raw_data.get("significant_associations", [])
        if not associations:
            return 0.1  # Very low confidence
        
        # Calculate average p-value and effect size
        avg_pvalue = sum(a["pValue"] for a in associations) / len(associations)
        max_effect = max(abs(a["beta"]) for a in associations)
        tissue_count = len(associations)
        
        # Multi-factor confidence scoring
        # P-value component (lower p = higher confidence)
        pvalue_score = max(0, 1 - (avg_pvalue * 10))  # Normalize p-values
        
        # Effect size component (larger effect = higher confidence)  
        effect_score = min(max_effect / 0.3, 1.0)  # Cap at 0.3 effect size
        
        # Tissue diversity component
        tissue_score = min(tissue_count / 3, 1.0)  # Max bonus for 3+ tissues
        
        # Weighted average (emphasize statistical significance)
        confidence = (0.5 * pvalue_score + 0.3 * effect_score + 0.2 * tissue_score)
        
        return max(0.05, min(0.95, confidence))  # Bound between 5% and 95%


class GnomADAPI(BiologicalAPITool):
    """
    gnomAD (Genome Aggregation Database) for population genetics data.
    
    Purpose: Get allele frequencies, constraint scores, predicted impact.
    API Documentation: https://gnomad.broadinstitute.org/api/
    """
    
    def construct_query(self, variant_id: str, gene_symbol: str) -> APIQuery:
        """
        PSEUDOCODE for gnomAD GraphQL query:
        
        Real implementation:
        POST https://gnomad.broadinstitute.org/api/
        Query: variant(variantId: $variantId) { allele_freq, consequence, ... }
        """
        return APIQuery(
            endpoint="https://gnomad.broadinstitute.org/api/",
            parameters={
                "query": f"""
                query {{
                    variant(variantId: "{variant_id}") {{
                        allele_freq
                        consequence
                        gene {{
                            symbol
                            constraint {{
                                pli
                                loeuf
                            }}
                        }}
                    }}
                }}
                """,
                "variables": {"variantId": variant_id}
            },
            expected_fields=["allele_freq", "consequence", "pli", "loeuf"],
            confidence_weight=0.9  # High trust in gnomAD constraint metrics
        )
    
    def parse_response(self, response: Dict) -> Evidence:
        """
        PSEUDOCODE for gnomAD data extraction:
        
        Key insights:
        - Rare variants (MAF < 0.01) more likely pathogenic
        - High pLI (>0.9) indicates loss-of-function intolerant gene
        - LOEUF < 0.35 suggests constraint against LoF variants
        """
        context = response.get("query", {})
        variant_id = context.get("variant_id", "").upper()
        gene_symbol = context.get("gene_symbol", "").upper()
        
        constrained_genes = {"MYBPC3", "MYH7", "TNNT2", "TPM1"}
        rare_variants = {"RS34357398"}
        
        allele_frequency = 0.002
        consequence = "synonymous_variant"
        pli = 0.55
        loeuf = 0.65
        impact = "MODERATE"
        
        if variant_id in rare_variants:
            allele_frequency = 0.0001
            consequence = "missense_variant"
            impact = "HIGH"
        
        if gene_symbol in constrained_genes:
            pli = 0.95
            loeuf = 0.28
        elif gene_symbol == "NOVEL_GENE":
            pli = 0.35
            loeuf = 0.82
            impact = "LOW"
        
        mock_constraint_data = {
            "allele_frequency": allele_frequency,
            "consequence": consequence,
            "gene_constraint": {
                "pli_score": pli,
                "loeuf_score": loeuf
            },
            "pathogenicity_indicators": {
                "is_rare": allele_frequency < 0.01,
                "in_constrained_gene": pli > 0.9,
                "predicted_impact": impact
            }
        }
        
        return Evidence(
            source_api="gnomAD",
            evidence_type="population_constraint",
            data=mock_constraint_data,
            confidence_score=self.calculate_confidence(mock_constraint_data),
            query_metadata={"version": "v3.1.2", "dataset": "genomes+exomes"}
        )
    
    def calculate_confidence(self, raw_data: Dict) -> float:
        """
        gnomAD confidence based on data completeness and constraint evidence.
        """
        # High confidence if variant is rare, gene is constrained, and impact predicted high
        is_rare = raw_data.get("allele_frequency", 1.0) < 0.01
        is_constrained = raw_data.get("gene_constraint", {}).get("pli_score", 0) > 0.9
        impact_level = raw_data.get("pathogenicity_indicators", {}).get("predicted_impact", "MODERATE")
        
        confidence = 0.4  # Base confidence
        if is_rare:
            confidence += 0.3
        if is_constrained:
            confidence += 0.2
        if impact_level == "HIGH":
            confidence += 0.1
        elif impact_level == "LOW":
            confidence -= 0.1
            
        return max(0.2, min(confidence, 0.9))


class UniProtAPI(BiologicalAPITool):
    """
    UniProt for protein function and disease association data.
    
    Purpose: Understand protein role, domains, known disease connections.
    API Documentation: https://rest.uniprot.org/
    """
    
    def construct_query(self, variant_id: str, gene_symbol: str) -> APIQuery:
        """
        PSEUDOCODE for UniProt REST API:
        
        Real query:
        GET https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_symbol}
        """
        return APIQuery(
            endpoint="https://rest.uniprot.org/uniprotkb/search",
            parameters={
                "query": f"gene:{gene_symbol} AND organism_id:9606",  # Human
                "format": "json",
                "fields": "accession,protein_name,function,disease"
            },
            expected_fields=["protein_name", "function", "disease_associations"],
            confidence_weight=0.7  # Moderate trust (manual curation varies)
        )
    
    def parse_response(self, response: Dict) -> Evidence:
        """
        PSEUDOCODE for UniProt function extraction:
        
        Focus on:
        - Protein functional domains
        - Known disease associations
        - Pathway involvement
        """
        context = response.get("query", {})
        gene_symbol = context.get("gene_symbol", "").upper()
        
        if gene_symbol == "MYBPC3":
            mock_protein_data = {
                "protein_name": "Cardiac myosin-binding protein C",
                "function_summary": "Regulates cardiac muscle contraction",
                "disease_associations": [
                    "Cardiomyopathy, hypertrophic familial",
                    "Cardiomyopathy, dilated"
                ],
                "domain_count": 7,
                "pathway_involvement": ["Muscle contraction", "Cardiac development"]
            }
        elif gene_symbol == "NOVEL_GENE":
            mock_protein_data = {
                "protein_name": "NOVEL_GENE protein",
                "function_summary": "",
                "disease_associations": [],
                "domain_count": 1,
                "pathway_involvement": []
            }
        else:
            mock_protein_data = {
                "protein_name": f"{gene_symbol or 'Unknown'} protein",
                "function_summary": "Function under investigation",
                "disease_associations": ["No curated disease association"],
                "domain_count": 3,
                "pathway_involvement": ["General cellular processes"]
            }
        
        return Evidence(
            source_api="UniProt",
            evidence_type="protein_function",
            data=mock_protein_data,
            confidence_score=self.calculate_confidence(mock_protein_data),
            query_metadata={"database": "Swiss-Prot", "manual_curation": True}
        )
    
    def calculate_confidence(self, raw_data: Dict) -> float:
        """
        UniProt confidence based on annotation completeness.
        """
        has_disease_info = len(raw_data.get("disease_associations", [])) > 0
        has_function_info = bool(raw_data.get("function_summary"))
        
        confidence = 0.35  # Base for protein existence
        if has_disease_info:
            confidence += 0.3
        if has_function_info:
            confidence += 0.25
        
        if raw_data.get("protein_name", "").startswith("Unknown"):
            confidence -= 0.1
            
        return max(0.2, min(confidence, 0.85))


# ============================================================================
# CORE AI AGENT: LLM-Powered Reasoning Engine
# ============================================================================

class V2GReasoningAgent:
    """
    Central AI agent that orchestrates tool calls and synthesizes evidence.
    
    Key Design Principles:
    1. Tool-agnostic: Can work with any BiologicalAPITool
    2. Evidence-based: Decisions based on aggregated confidence scores
    3. Transparent: Clear reasoning chain for interpretability
    4. Self-verifying: Built-in checks for consistency and quality
    """
    
    def __init__(self, tools: List[BiologicalAPITool], llm_client=None):
        """
        Initialize agent with biological data tools and optional LLM.
        
        Args:
            tools: List of API tools (GTEx, gnomAD, UniProt, etc.)
            llm_client: Optional LLM for text synthesis (OpenAI, Claude, etc.)
        """
        self.tools = {tool.__class__.__name__.replace('API', '').lower(): tool for tool in tools}
        self.llm_client = llm_client
        self.reasoning_chain = []  # Track decision-making process
    
    def analyze_variant_gene_relationship(self, variant_id: str, gene_symbol: str, 
                                         disease_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Main analysis pipeline implementing agentic reasoning workflow.
        
        AGENT REASONING CHAIN:
        1. PLAN: Determine which tools to use based on context
        2. QUERY: Execute API calls to gather evidence  
        3. AGGREGATE: Combine evidence with confidence weighting
        4. VERIFY: Check for consistency and conflicts
        5. SYNTHESIZE: Generate human-readable summary
        
        Returns:
            Complete analysis with evidence, scores, and reasoning
        """
        
        # STEP 1: PLANNING PHASE
        print(f"\n=== AGENT ANALYSIS: {variant_id} ↔ {gene_symbol} ===")
        self.reasoning_chain = []
        
        analysis_strategy = self._plan_analysis_strategy(variant_id, gene_symbol, disease_context)
        self.reasoning_chain.append(f"Selected strategy: {analysis_strategy['approach']}")
        
        print(f"Strategy: {analysis_strategy['approach']}")
        print(f"Priority Tools: {analysis_strategy['priority_tools']}")
        
        # STEP 2: EVIDENCE COLLECTION PHASE
        print("\n--- Evidence Collection ---")
        evidence_list = []
        
        for tool_name in analysis_strategy['priority_tools']:
            if tool_name in self.tools:
                print(f"Querying {tool_name.upper()}...")
                
                # PSEUDOCODE: Real implementation would make HTTP requests
                tool = self.tools[tool_name]
                query = tool.construct_query(variant_id, gene_symbol)
                
                # Mock response - real version would: response = requests.post(query.endpoint, query.parameters)
                mock_response = {
                    "status": "success",
                    "query": {
                        "variant_id": variant_id,
                        "gene_symbol": gene_symbol
                    }
                }
                
                evidence = tool.parse_response(mock_response)
                evidence_list.append(evidence)
                
                print(f"  → Evidence collected (confidence: {evidence.confidence_score:.2f})")
                self.reasoning_chain.append(f"{tool_name}: confidence {evidence.confidence_score:.2f}")
        
        # STEP 3: EVIDENCE AGGREGATION PHASE
        print("\n--- Evidence Aggregation ---")
        overall_score = self._calculate_relationship_score(evidence_list, analysis_strategy)
        print(f"Overall Relationship Score: {overall_score:.2f}/1.00")
        
        # STEP 4: SELF-VERIFICATION PHASE
        print("\n--- Self-Verification ---")
        verification_result = self._verify_evidence_consistency(evidence_list)
        print(f"Verification Status: {verification_result['status']}")
        
        if verification_result['warnings']:
            for warning in verification_result['warnings']:
                print(f"  Warning: {warning}")
        
        # STEP 5: LLM SYNTHESIS PHASE (if available)
        print("\n--- Summary Generation ---")
        summary = self._generate_summary(variant_id, gene_symbol, evidence_list, 
                                       overall_score, disease_context)
        
        return {
            "variant_id": variant_id,
            "gene_symbol": gene_symbol,
            "relationship_score": overall_score,
            "evidence": evidence_list,
            "verification": verification_result,
            "summary": summary,
            "reasoning_chain": self.reasoning_chain.copy()
        }
    
    def _plan_analysis_strategy(self, variant_id: str, gene_symbol: str, 
                               disease_context: Optional[str]) -> Dict[str, Any]:
        """
        AGENTIC PLANNING: Choose optimal analysis strategy based on context.
        
        This demonstrates AI reasoning by adapting approach based on:
        - Disease context (cardiac vs. neurological vs. cancer)
        - Gene characteristics (known disease gene vs. novel)
        - Variant type (common vs. rare)
        """
        
        # Default strategy
        strategy = {
            "approach": "comprehensive",
            "priority_tools": ["gtex", "gnomad", "uniprot"],
            "evidence_weights": {"gtex": 0.4, "gnomad": 0.4, "uniprot": 0.2}
        }
        
        # ADAPTIVE REASONING: Modify strategy based on context
        if disease_context and "cardio" in disease_context.lower():
            # For cardiac diseases, prioritize tissue-specific expression
            strategy["approach"] = "tissue_focused"
            strategy["evidence_weights"] = {"gtex": 0.5, "gnomad": 0.3, "uniprot": 0.2}
            self.reasoning_chain.append("Cardiac context detected: prioritizing tissue expression")
        
        # ADAPTIVE REASONING: Known vs. novel genes
        known_disease_genes = {"MYBPC3", "BRCA1", "BRCA2", "TP53", "LDLR"}
        if gene_symbol in known_disease_genes:
            strategy["approach"] = "validation_focused"
            strategy["evidence_weights"] = {"gnomad": 0.5, "gtex": 0.3, "uniprot": 0.2}
            self.reasoning_chain.append(f"Known disease gene {gene_symbol}: focusing on constraint evidence")
        
        return strategy
    
    def _calculate_relationship_score(self, evidence_list: List[Evidence], 
                                     strategy: Dict[str, Any]) -> float:
        """
        EVIDENCE AGGREGATION with adaptive weighting.
        
        Combines evidence from multiple sources using strategy-specific weights.
        This is a key AI/ML concept: weighted ensemble prediction.
        """
        if not evidence_list:
            return 0.0
        
        # Extract strategy weights
        weights = strategy.get("evidence_weights", {})
        
        # Calculate weighted average confidence
        total_weight = 0
        weighted_sum = 0
        
        for evidence in evidence_list:
            source_name = evidence.source_api.lower()
            weight = weights.get(source_name, 0.33)  # Default equal weighting
            
            weighted_sum += evidence.confidence_score * weight
            total_weight += weight
        
        base_score = weighted_sum / total_weight if total_weight > 0 else 0
        
        coverage_ratio = min(1.0, len(evidence_list) / max(1, len(self.tools)))
        consistency_bonus = 0.05 if (len(evidence_list) >= 3 and base_score >= 0.6) else 0.02 if len(evidence_list) >= 2 else 0.0
        enhanced_score = base_score + consistency_bonus
        enhanced_score *= 0.7 + 0.3 * coverage_ratio
        
        if consistency_bonus > 0:
            self.reasoning_chain.append("Evidence consistency bonus applied")
        
        return min(enhanced_score, 0.95)
    
    def _verify_evidence_consistency(self, evidence_list: List[Evidence]) -> Dict[str, Any]:
        """
        SELF-VERIFICATION: Check for conflicts and data quality issues.
        
        This implements the "chain-of-verification" pattern from recent AI research.
        """
        verification = {
            "status": "VERIFIED",
            "warnings": [],
            "confidence_level": "HIGH"
        }
        
        # Check 1: Minimum evidence threshold
        if len(evidence_list) < 2:
            verification["warnings"].append("Limited evidence sources (recommend ≥2 for confidence)")
            verification["confidence_level"] = "LOW"
        
        # Check 2: Confidence score distribution
        confidence_scores = [e.confidence_score for e in evidence_list]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        if avg_confidence < 0.5:
            verification["warnings"].append("Low average confidence scores")
            verification["confidence_level"] = "MEDIUM"
        
        # Check 3: Source diversity
        sources = set(e.source_api for e in evidence_list)
        if len(sources) < len(evidence_list):
            verification["warnings"].append("Duplicate evidence sources detected")
        
        # Update overall status
        if verification["warnings"]:
            verification["status"] = "VERIFIED_WITH_WARNINGS"
        
        return verification
    
    def _generate_summary(self, variant_id: str, gene_symbol: str, evidence_list: List[Evidence],
                         score: float, disease_context: Optional[str]) -> str:
        """
        SUMMARY GENERATION: Convert technical evidence to human-readable report.
        
        In real implementation, this would use LLM API:
        - OpenAI GPT-4 for medical text generation
        - Claude for scientific writing
        - Local models for privacy-sensitive data
        """
        
        if self.llm_client:
            # PSEUDOCODE for LLM integration
            prompt = f"""
            Analyze the variant-gene relationship based on this evidence:
            
            Variant: {variant_id}
            Gene: {gene_symbol}
            Disease Context: {disease_context or 'General'}
            Overall Score: {score:.2f}
            
            Evidence Summary:
            {self._format_evidence_for_llm(evidence_list)}
            
            Generate a concise scientific summary explaining:
            1. Strength of evidence for relationship
            2. Key supporting findings
            3. Confidence level and limitations
            4. Recommendations for validation
            """
            
            # Mock LLM response - real implementation: response = llm_client.complete(prompt)
            llm_summary = f"""
            ANALYSIS SUMMARY for {variant_id} ↔ {gene_symbol}:
            
            Evidence Level: {'Strong' if score > 0.7 else 'Moderate' if score > 0.4 else 'Weak'}
            
            Key Findings:
            • Expression data suggests tissue-specific effects in relevant cell types
            • Population genetics indicates {'' if score > 0.5 else 'limited '}constraint evidence
            • Protein function analysis provides {'direct' if score > 0.6 else 'indirect'} disease relevance
            
            Confidence: {score:.1%} based on {len(evidence_list)} evidence sources
            
            Recommendation: {'High priority for experimental validation' if score > 0.7 else 'Consider additional evidence before validation'}
            """
            
            return llm_summary
        
        else:
            # Fallback: Rule-based summary
            confidence_level = "High" if score > 0.7 else "Medium" if score > 0.4 else "Low"
            
            evidence_labels = ", ".join([e.evidence_type for e in evidence_list]) or "no evidence collected"
            context_note = disease_context or "No disease context provided"
            rationale = (
                "Strong evidence supports"
                if score > 0.7 else
                "Moderate evidence suggests"
                if score > 0.4 else
                "Limited evidence for"
            )
            coverage = len(evidence_list)
            
            lines = [
                f"Relationship Analysis: {variant_id} ↔ {gene_symbol}",
                f"Confidence Level: {confidence_level} ({score:.2f}/1.00)",
                f"Evidence Sources: {coverage}",
                f"Context: {context_note}",
                f"{rationale} a functional relationship between this variant and gene.",
                f"Primary evidence: {evidence_labels}"
            ]
            
            if score < 0.5:
                lines.append("Recommendation: gather additional orthogonal evidence before prioritizing experiments.")
            elif score < 0.7:
                lines.append("Recommendation: supportive evidence exists; consider targeted validation assays.")
            else:
                lines.append("Recommendation: prioritize for experimental follow-up (e.g., CRISPR or iPSC models).")
            
            return "\n".join(lines)
    
    def _format_evidence_for_llm(self, evidence_list: List[Evidence]) -> str:
        """
        Format evidence for LLM prompt engineering.
        
        WHY STRUCTURED FORMATTING: LLMs perform better with consistent,
        well-formatted input. This creates standardized evidence summaries
        that enable reliable text generation.
        
        Args:
            evidence_list: List of Evidence objects from different sources
            
        Returns:
            Formatted string with source, type, and confidence for each evidence
        """
        formatted = []
        for evidence in evidence_list:
            formatted.append(f"- {evidence.source_api}: {evidence.evidence_type} (confidence: {evidence.confidence_score:.2f})")
        return "\n".join(formatted)


# ============================================================================
# DEMONSTRATION WORKFLOW
# ============================================================================

def main():
    """
    Demonstration of the conceptual V2G agentic workflow.
    
    This shows the DESIGN PATTERN rather than full implementation,
    focusing on the AI/ML concepts requested in the assessment.
    
    DESIGN DECISION: Using real variant data from sample_variants.csv
    to demonstrate how the agent would work with actual genomic data.
    """
    
    print("=== V2G AGENTIC WORKFLOW DEMONSTRATION ===")
    print("Conceptual design focusing on AI agent orchestration\n")
    
    # Initialize biological data tools
    tools = [
        GTExAPI(),      # Gene expression eQTLs
        GnomADAPI(),    # Population constraint data  
        UniProtAPI()    # Protein function information
    ]
    
    # Initialize reasoning agent
    # In real implementation: agent = V2GReasoningAgent(tools, llm_client=OpenAI(api_key="..."))
    agent = V2GReasoningAgent(tools, llm_client=None)  # Using fallback summary
    
    # Load real variant data for demonstration
    try:
        df = pd.read_csv("sample_variants.csv")
        print(f"Loaded {len(df)} variants from sample_variants.csv")
        
        # Select interesting variants for analysis
        test_cases = [
            {
                "variant_id": "rs34357398",  # Known HCM variant from data
                "gene_symbol": "MYBPC3", 
                "disease_context": "Hypertrophic Cardiomyopathy"
            },
            {
                "variant_id": df.iloc[0]['variant_id'],  # First variant from real data
                "gene_symbol": "NOVEL_GENE",
                "disease_context": None
            }
        ]
        
    except FileNotFoundError:
        print("Warning: sample_variants.csv not found, using mock examples")
        # Fallback to hardcoded examples
        test_cases = [
            {
                "variant_id": "rs34357398",
                "gene_symbol": "MYBPC3", 
                "disease_context": "Hypertrophic Cardiomyopathy"
            },
            {
                "variant_id": "rs123456",
                "gene_symbol": "NOVEL_GENE",
                "disease_context": None
            }
        ]
    
    results = []
    for case in test_cases:
        result = agent.analyze_variant_gene_relationship(
            variant_id=case["variant_id"],
            gene_symbol=case["gene_symbol"],
            disease_context=case["disease_context"]
        )
        results.append(result)
        
        print("\nSUMMARY:")
        print(result["summary"])
        print(f"\nReasoning Chain: {' → '.join(result['reasoning_chain'])}")
        print("\n" + "="*60)
    
    print(f"\nAnalyzed {len(results)} variant-gene relationships")
    print("Demonstration complete - see design patterns above")


if __name__ == "__main__":
    main()
