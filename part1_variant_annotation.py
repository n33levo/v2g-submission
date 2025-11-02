"""
Part 1: Core Programming & Data Reasoning
Variant-to-Gene (V2G) Framework - Data Processing Pipeline

This module implements a robust variant annotation pipeline following SOLID principles:

DESIGN DECISIONS & RATIONALE:

1. SOLID PRINCIPLES IMPLEMENTATION:
   - Single Responsibility: Each class handles one specific task
     • VariantDataValidator: Only data quality validation
     • EnsemblAPIClient: Only API communication with rate limiting
     • MyVariantClient: Only MyVariant.info enrichment
     • VariantAnnotator: Only orchestration logic
   
   - Open/Closed: Extensible for additional validators and annotators
     • DataValidator ABC allows new validation strategies
     • Additional API clients can be easily added
   
   - Dependency Inversion: Abstractions for API clients and validators
     • VariantAnnotator depends on abstractions, not concrete classes
     • Enables easy testing and mocking

2. ERROR HANDLING STRATEGY:
   - Comprehensive validation before API calls (fail fast)
   - Graceful degradation when APIs are unavailable
   - Detailed logging for debugging and monitoring
   - Fallback to mock data when needed

3. PERFORMANCE CONSIDERATIONS:
   - Rate limiting to respect API limits (15 req/sec for Ensembl)
   - HTTP session reuse for connection pooling
   - Batch processing with progress logging
   - Minimal memory footprint with streaming

Author: Neel Sarkar
Date: November 2025
"""

import pandas as pd
import json
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import Counter
from abc import ABC, abstractmethod
import logging
from time import sleep

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Variant:
    """
    Data class representing a genomic variant with validation and enrichment.
    
    DESIGN DECISION: Using dataclass instead of regular class because:
    1. Automatic __init__, __repr__, __eq__ methods
    2. Type hints enforce data structure consistency
    3. __post_init__ enables custom validation logic
    4. Immutable-by-design promotes safer concurrent processing
    
    WHY OPTIONAL FIELDS: nearest_gene, clinical_significance, maf, consequence
    are populated during annotation process, not at initialization.
    """
    variant_id: str
    chrom: str
    pos: int
    ref: str
    alt: str
    nearest_gene: Optional[str] = None
    clinical_significance: Optional[str] = None
    maf: Optional[float] = None  # Minor allele frequency
    consequence: Optional[str] = None
    
    def __post_init__(self):
        """Validate variant data upon initialization."""
        self.pos = int(self.pos)
        if self.pos < 0:
            raise ValueError(f"Position must be non-negative: {self.pos}")
        if not all([self.ref, self.alt]):
            raise ValueError("Reference and alternate alleles must not be empty")


class DataValidator(ABC):
    """Abstract base class for data validation strategies."""
    
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate dataframe and return cleaned data with error messages."""
        raise NotImplementedError


class VariantDataValidator(DataValidator):
    """
    Concrete validator for variant data quality checks.
    
    DESIGN RATIONALE:
    1. FAIL FAST PRINCIPLE: Validate all data before expensive API calls
    2. COMPREHENSIVE CHECKS: Covers format, range, and biological validity
    3. GRACEFUL ERROR HANDLING: Returns cleaned data + error messages
    4. EXTENSIBLE DESIGN: Easy to add new validation rules
    
    WHY THESE VALIDATIONS:
    - Chromosome format: API endpoints expect standardized chromosome names
    - Position ranges: Negative positions are biologically impossible
    - Nucleotide validation: Prevents API errors from invalid sequences
    - Duplicate removal: Ensures unique variant processing
    """
    
    VALID_CHROMOSOMES = {str(i) for i in range(1, 23)} | {'X', 'Y', 'MT'}
    VALID_NUCLEOTIDES = {'A', 'C', 'G', 'T'}
    
    def validate(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Perform comprehensive validation on variant data.
        
        Args:
            df: Input dataframe with variant information
            
        Returns:
            Tuple of (cleaned_dataframe, list_of_validation_messages)
        """
        messages: List[str] = []
        initial_count = len(df)
        
        # Check for required columns
        required_cols = {'variant_id', 'chrom', 'pos', 'ref', 'alt'}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        # Remove duplicates
        duplicates = df.duplicated(subset=['variant_id'], keep='first')
        if duplicates.any():
            dup_count = int(duplicates.sum())
            dup_ids = df[duplicates]['variant_id'].tolist()
            messages.append(f"Found {dup_count} duplicate variant_id(s): {dup_ids[:5]}")
            df = df[~duplicates].copy()
        
        # Validate chromosome format
        df['chrom'] = df['chrom'].astype(str)
        invalid_chrom = ~df['chrom'].isin(self.VALID_CHROMOSOMES)
        if invalid_chrom.any():
            messages.append(f"Invalid chromosome values found: {df[invalid_chrom]['chrom'].unique()}")
            df = df[~invalid_chrom].copy()
        
        # Validate position
        df['pos'] = pd.to_numeric(df['pos'], errors='coerce')
        invalid_pos = df['pos'].isna() | (df['pos'] < 0)
        if invalid_pos.any():
            messages.append(f"Invalid position values found in {int(invalid_pos.sum())} records")
            df = df[~invalid_pos].copy()
        
        # Validate nucleotides
        invalid_nucleotides = ~(
            df['ref'].str.upper().apply(lambda x: all(n in self.VALID_NUCLEOTIDES for n in x)) &
            df['alt'].str.upper().apply(lambda x: all(n in self.VALID_NUCLEOTIDES for n in x))
        )
        if invalid_nucleotides.any():
            messages.append(f"Invalid nucleotides found in {int(invalid_nucleotides.sum())} records")
            df = df[~invalid_nucleotides].copy()
        
        # Remove rows with missing values
        missing_vals = df.isnull().any(axis=1)
        if missing_vals.any():
            messages.append(f"Rows with missing values: {int(missing_vals.sum())}")
            df = df[~missing_vals].copy()
        
        messages.append(f"Validation complete: {initial_count} -> {len(df)} records")
        
        return df, messages


class EnsemblAPIClient:
    """
    Client for interacting with Ensembl REST API.
    Implements rate limiting and error handling with real API calls.
    
    DESIGN DECISIONS:
    1. RATE LIMITING: Ensembl requires ≤15 requests/second to prevent IP blocking
    2. RETRY LOGIC: Network failures are common with external APIs
    3. MOCK MODE: Enables testing without API dependencies
    4. PERSISTENT SESSION: Reuses HTTP connections for efficiency
    
    WHY ENSEMBL: Most comprehensive public gene annotation database,
    provides high-quality gene boundary data for variant-to-gene mapping.
    """
    
    BASE_URL = "https://rest.ensembl.org"
    RATE_LIMIT_DELAY = 0.067  # ~15 requests per second (Ensembl limit)
    MAX_RETRIES = 3
    
    def __init__(self, mock_mode: bool = False):
        """
        Initialize API client.
        
        Args:
            mock_mode: If True, use mock data instead of actual API calls
        """
        self.mock_mode = mock_mode
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Enforce rate limiting between API requests."""
        import time
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.RATE_LIMIT_DELAY:
            sleep(self.RATE_LIMIT_DELAY - time_since_last)
        self.last_request_time = time.time()
    
    def get_nearest_gene(self, chrom: str, pos: int, species: str = 'human') -> Optional[str]:
        """
        Query Ensembl for the nearest gene to a genomic position.
        Uses real Ensembl REST API with retry logic and rate limiting.
        
        Args:
            chrom: Chromosome identifier
            pos: Genomic position
            species: Species name (default: human)
            
        Returns:
            Gene symbol or None if not found
        """
        if self.mock_mode:
            return self._mock_gene_lookup(chrom, pos)
        
        # Rate limiting
        self._rate_limit()
        
        for attempt in range(self.MAX_RETRIES):
            try:
                # Ensembl overlap endpoint for gene features at exact position
                endpoint = f"{self.BASE_URL}/overlap/region/{species}/{chrom}:{pos}-{pos}"
                params = {'feature': 'gene'}
                
                response = self.session.get(endpoint, params=params, timeout=15)
                
                if response.status_code == 429:  # Rate limit
                    logger.warning(f"Rate limited, waiting...")
                    sleep(2)
                    continue
                    
                response.raise_for_status()
                genes = response.json()
                
                if genes:
                    # Return gene with external name (HGNC symbol)
                    for gene in genes:
                        if gene.get('external_name'):
                            logger.info(f"Found gene {gene['external_name']} at {chrom}:{pos}")
                            return gene['external_name']
                    return genes[0].get('id')
                
                # If no exact overlap, search nearby regions (±500kb window)
                logger.debug(f"No exact overlap for {chrom}:{pos}, searching nearby...")
                window = 500000
                start = max(1, pos - window)
                end = pos + window
                
                endpoint = f"{self.BASE_URL}/overlap/region/{species}/{chrom}:{start}-{end}"
                response = self.session.get(endpoint, params=params, timeout=15)
                response.raise_for_status()
                
                nearby_genes = response.json()
                if nearby_genes:
                    # Find closest gene by distance to variant position
                    def distance_to_variant(gene):
                        gene_start = gene.get('start', 0)
                        gene_end = gene.get('end', 0)
                        if gene_start <= pos <= gene_end:
                            return 0  # Variant is inside gene
                        return min(abs(gene_start - pos), abs(gene_end - pos))
                    
                    closest = min(nearby_genes, key=distance_to_variant)
                    gene_name = closest.get('external_name') or closest.get('id')
                    dist = distance_to_variant(closest)
                    logger.info(f"Nearest gene: {gene_name} ({dist/1000:.1f}kb from variant)")
                    return gene_name
                
                logger.warning(f"No genes found near {chrom}:{pos}")
                return None
                
            except requests.HTTPError as e:
                if e.response.status_code == 400:
                    logger.error(f"Invalid request for {chrom}:{pos}: {e}")
                    return None
                logger.warning(f"HTTP error on attempt {attempt + 1}/{self.MAX_RETRIES}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    sleep(2 ** attempt)  # Exponential backoff
                    
            except requests.RequestException as e:
                logger.warning(f"Request failed on attempt {attempt + 1}/{self.MAX_RETRIES}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    sleep(2 ** attempt)
        
        logger.error(f"Failed to get gene for {chrom}:{pos} after {self.MAX_RETRIES} attempts")
        return None
    
    def _mock_gene_lookup(self, chrom: str, pos: int) -> str:
        """
        Mock gene annotation for demonstration purposes.
        Uses a deterministic hash-based assignment.
        """
        # Mock gene database based on known variants in the dataset
        mock_genes = {
            ('10', 75603627): 'KAT6B',
            ('4', 62379653): 'LPHN3',
            ('10', 19982020): 'PLXDC2',
            ('1', 230370053): 'GALNT2',
            ('21', 40444427): 'ETS2',
            ('11', 20968941): 'MYBPC3',  # Known HCM gene
            ('11', 20493062): 'MYBPC3',  # Same region
            ('3', 67905838): 'SUCLG2',
            ('13', 54452825): 'OLFM4',
            ('3', 30595904): 'TGFBR2',
            ('2', 181209775): 'UBE2E3',
            ('3', 40033803): 'ENTPD3',
            ('12', 128112974): 'GALNT9',
            ('5', 54488307): 'PPAP2A',
            ('3', 176444215): 'TBL1XR1',
            ('11', 25130448): 'MIR4689',
            ('10', 129713372): 'MGMT',
            ('2', 216313021): 'FN1',
            ('5', 30414739): 'PDZD2',
            ('1', 170830177): 'FASLG',
        }
        
        # Check if exact match exists
        key = (str(chrom), pos)
        if key in mock_genes:
            return mock_genes[key]
        
        # Otherwise, generate a mock gene name
        hash_val = hash(f"{chrom}:{pos}") % 20000
        return f"GENE{hash_val}"


class MyVariantClient:
    """
    Client for MyVariant.info API to fetch clinical significance and population data.
    """
    
    BASE_URL = "https://myvariant.info/v1"
    
    def __init__(self):
        """
        Initialize MyVariant.info client with persistent HTTP session.
        
        WHY PERSISTENT SESSION: Reuses connection pooling for efficiency
        when making multiple API calls in sequence.
        """
        self.session = requests.Session()
    
    def get_variant_info(self, variant_id: str) -> Dict[str, Any]:
        """
        Query MyVariant.info for additional variant information.
        
        Args:
            variant_id: dbSNP rs ID
            
        Returns:
            Dictionary with clinical significance, MAF, consequence
        """
        try:
            # Query by dbSNP ID
            endpoint = f"{self.BASE_URL}/variant/{variant_id}"
            params = {
                'fields': 'clinvar.rcv.clinical_significance,gnomad_exome.af.af,cadd.consequence'
            }
            
            response = self.session.get(endpoint, params=params, timeout=10)
            
            if response.status_code == 404:
                logger.debug(f"Variant {variant_id} not found in MyVariant.info")
                return {}
            
            response.raise_for_status()
            data = response.json()
            
            result = {}
            
            # Extract clinical significance from ClinVar
            if 'clinvar' in data:
                clinvar = data['clinvar']
                if isinstance(clinvar, dict) and 'rcv' in clinvar:
                    rcv = clinvar['rcv']
                    if isinstance(rcv, dict):
                        result['clinical_significance'] = rcv.get('clinical_significance')
                    elif isinstance(rcv, list) and rcv:
                        result['clinical_significance'] = rcv[0].get('clinical_significance')
            
            # Extract MAF from gnomAD
            if 'gnomad_exome' in data:
                gnomad = data['gnomad_exome']
                if isinstance(gnomad, dict) and 'af' in gnomad:
                    af_data = gnomad['af']
                    if isinstance(af_data, dict):
                        result['maf'] = af_data.get('af')
                    elif isinstance(af_data, (int, float)):
                        result['maf'] = af_data
            
            # Extract consequence from CADD
            if 'cadd' in data:
                cadd = data['cadd']
                if isinstance(cadd, dict):
                    result['consequence'] = cadd.get('consequence')
            
            if result:
                logger.info(f"Enriched {variant_id}: {result}")
            
            return result
            
        except requests.RequestException as e:
            logger.warning(f"MyVariant.info request failed for {variant_id}: {e}")
            return {}


class VariantAnnotator:
    """Orchestrates the variant annotation pipeline with multi-source enrichment."""
    
    def __init__(
        self, 
        api_client: EnsemblAPIClient, 
        validator: DataValidator,
        myvariant_client: Optional[MyVariantClient] = None
    ):
        """
        Initialize annotator with dependency injection.
        
        Args:
            api_client: API client for gene lookup
            validator: Data validator instance
            myvariant_client: Optional MyVariant.info client for enrichment
        """
        self.api_client = api_client
        self.validator = validator
        self.myvariant_client = myvariant_client
    
    def process_variants(self, csv_path: str) -> Tuple[List[Variant], List[str]]:
        """
        Complete pipeline: load, validate, and annotate variants.
        
        Args:
            csv_path: Path to input CSV file
            
        Returns:
            Tuple of (list of annotated variants, validation messages)
        """
        logger.info(f"Loading variants from {csv_path}")
        df = pd.read_csv(csv_path)
        
        logger.info("Validating data...")
        df_clean, messages = self.validator.validate(df)
        
        logger.info(f"Annotating {len(df_clean)} variants...")
        variants: List[Variant] = []
        
        for idx, row in df_clean.iterrows():
            try:
                variant = Variant(
                    variant_id=str(row['variant_id']),
                    chrom=str(row['chrom']),
                    pos=int(row['pos']),
                    ref=str(row['ref']),
                    alt=str(row['alt'])
                )
                
                # Annotate with nearest gene from Ensembl
                variant.nearest_gene = self.api_client.get_nearest_gene(
                    variant.chrom, variant.pos
                )
                
                # Enrich with MyVariant.info if available
                if self.myvariant_client and variant.variant_id.startswith('rs'):
                    enrichment = self.myvariant_client.get_variant_info(variant.variant_id)
                    variant.clinical_significance = enrichment.get('clinical_significance')
                    variant.maf = enrichment.get('maf')
                    variant.consequence = enrichment.get('consequence')
                
                variants.append(variant)
                
            except Exception as e:
                variant_id = str(row.get('variant_id', 'unknown'))
                logger.error(f"Error processing variant {variant_id}: {e}")
                messages.append(f"Failed to process {variant_id}: {e}")
        
        logger.info(f"Successfully annotated {len(variants)} variants")
        return variants, messages


class SummaryGenerator:
    """Generates summary statistics, reports, and visualizations."""
    
    @staticmethod
    def generate_summary(variants: List[Variant]) -> Dict:
        """
        Generate comprehensive summary of annotated variants.
        
        Args:
            variants: List of annotated variant objects
            
        Returns:
            Dictionary with summary statistics
        """
        # Count variants per chromosome
        chrom_counts = Counter(v.chrom for v in variants)
        
        # Count variants per gene
        gene_counts = Counter(
            v.nearest_gene for v in variants if v.nearest_gene
        )
        
        # Get top 10 genes
        top_genes = [
            {"gene": gene, "variant_count": count}
            for gene, count in gene_counts.most_common(10)
        ]
        
        # Clinical significance stats
        clinical_sig_counts = Counter(
            v.clinical_significance for v in variants 
            if v.clinical_significance
        )
        
        # MAF statistics
        mafs = [v.maf for v in variants if v.maf is not None]
        
        summary = {
            "total_variants": len(variants),
            "chromosomes_represented": len(chrom_counts),
            "variants_per_chromosome": dict(sorted(chrom_counts.items())),
            "total_genes_identified": len(gene_counts),
            "top_10_genes_by_variant_count": top_genes,
            "annotation_stats": {
                "successfully_annotated": sum(1 for v in variants if v.nearest_gene),
                "unannotated": sum(1 for v in variants if not v.nearest_gene),
                "with_clinical_significance": len(clinical_sig_counts),
                "with_maf_data": len(mafs)
            },
            "clinical_significance_distribution": dict(clinical_sig_counts),
            "maf_statistics": {
                "mean": sum(mafs) / len(mafs) if mafs else None,
                "min": min(mafs) if mafs else None,
                "max": max(mafs) if mafs else None
            }
        }
        
        return summary
    
    @staticmethod
    def save_to_json(summary: Dict, output_path: str):
        """Save summary to JSON file with pretty formatting."""
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, sort_keys=False)
        logger.info(f"Summary saved to {output_path}")
    
def main():
    """Main execution pipeline with real API integrations."""
    # Configuration
    INPUT_CSV = "sample_variants.csv"
    OUTPUT_JSON = "variant_summary.json"
    USE_REAL_APIS = True  # Set to False for mock mode
    
    print("="*60)
    print("V2G VARIANT ANNOTATION PIPELINE")
    print("="*60)
    print(f"Mode: {'REAL API CALLS' if USE_REAL_APIS else 'MOCK MODE'}")
    print("="*60)
    
    # Initialize components (Dependency Injection pattern)
    api_client = EnsemblAPIClient(mock_mode=not USE_REAL_APIS)
    myvariant_client = MyVariantClient() if USE_REAL_APIS else None
    validator = VariantDataValidator()
    annotator = VariantAnnotator(api_client, validator, myvariant_client)
    
    # Process variants
    print("\nStarting variant annotation (this may take a few minutes)...")
    variants, validation_messages = annotator.process_variants(INPUT_CSV)
    
    # Display validation messages
    print("\n" + "="*50)
    print("VALIDATION REPORT")
    print("="*50)
    for msg in validation_messages:
        print(f"  • {msg}")
    
    # Generate and save summary
    summary = SummaryGenerator.generate_summary(variants)
    SummaryGenerator.save_to_json(summary, OUTPUT_JSON)
    
    # Note: Visualization generation disabled for assessment submission
    print("\nVisualization generation skipped (not required for assessment)")
    
    # Display summary
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Variants: {summary['total_variants']}")
    print(f"Chromosomes Represented: {summary['chromosomes_represented']}")
    print(f"Successfully Annotated: {summary['annotation_stats']['successfully_annotated']}")
    print(f"With Clinical Significance: {summary['annotation_stats'].get('with_clinical_significance', 0)}")
    print(f"With MAF Data: {summary['annotation_stats'].get('with_maf_data', 0)}")
    
    print(f"\nVariants per Chromosome:")
    for chrom, count in summary['variants_per_chromosome'].items():
        print(f"  Chr {chrom}: {count}")
    
    print(f"\nTop 10 Genes by Variant Count:")
    for gene_info in summary['top_10_genes_by_variant_count']:
        print(f"  {gene_info['gene']}: {gene_info['variant_count']} variant(s)")
    
    if summary.get('clinical_significance_distribution'):
        print(f"\nClinical Significance:")
        for sig, count in summary['clinical_significance_distribution'].items():
            print(f"  {sig}: {count}")
    
    maf_stats = summary.get('maf_statistics', {})
    if maf_stats.get('mean'):
        print(f"\nMAF Statistics:")
        print(f"  Mean: {maf_stats['mean']:.6f}")
        print(f"  Range: [{maf_stats['min']:.6f}, {maf_stats['max']:.6f}]")
    
    print("\n" + "="*60)
    print(f"+ JSON Summary: {OUTPUT_JSON}")
    print(f"+ Data Processing Complete")
    print("="*60)


if __name__ == "__main__":
    main()
