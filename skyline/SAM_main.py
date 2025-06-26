# query_skyline.py
import os
import json
import argparse
import numpy as np
import tqdm
from SAM_utils import setup_sam_predictor, extract_skyline_from_image, compare_skylines_fft

def load_database(database_path):
    """Load the skyline database from JSON file."""
    with open(database_path, 'r') as f:
        database = json.load(f)
    
    # Convert skyline vectors back to numpy arrays
    for entry in database:
        entry['skyline_vector'] = np.array(entry['skyline_vector'])
    
    return database

def query_skyline_database(query_images, database, predictor, top_k=5):
    """
    Query the skyline database with multiple images from a folder.
    
    Args:
        query_images (list): List of query image paths
        database (list): List of database entries
        predictor: SAM predictor object
        top_k (int): Number of top matches to return per query
    
    Returns:
        dict: Results for each query image
    """
    all_results = {}
    
    print(f"Processing {len(query_images)} query images...")
    
    for query_image_path in tqdm(query_images, desc="Processing queries"):
        query_filename = os.path.basename(query_image_path)
        print(f"\nExtracting skyline from: {query_filename}")
        
        query_skyline = extract_skyline_from_image(query_image_path, predictor)
        
        if query_skyline is None:
            print(f"Failed to extract skyline from {query_filename}")
            all_results[query_filename] = []
            continue
        
        results = []
        for i, db_entry in enumerate(database):
            db_skyline = db_entry['skyline_vector']
            
            # Compare skylines using FFT-based correlation
            score, rotation = compare_skylines_fft(query_skyline, db_skyline)
            
            result = {
                'database_index': i,
                'image_path': db_entry['image_path'],
                'gps': db_entry['gps'],
                'correlation_score': score,
                'optimal_rotation': rotation,
                'skyline_vector': db_skyline
            }
            results.append(result)
        
        # Sort by correlation score (highest first)
        results.sort(key=lambda x: x['correlation_score'], reverse=True)
        all_results[query_filename] = results[:top_k]
    
    return all_results

def print_results(all_results):
    """Print the query results for all images in a formatted way."""
    print(f"\n{'='*80}")
    print(f"BATCH QUERY RESULTS")
    print(f"{'='*80}")
    
    if not all_results:
        print("No results found.")
        return
    
    for query_filename, results in all_results.items():
        print(f"\nQUERY: {query_filename}")
        print("-" * 80)
        
        if not results:
            print("  No matches found for this image.")
            continue
        
        print(f"  Top {len(results)} matches:")
        for i, result in enumerate(results, 1):
            print(f"  {i:2d}. Score: {result['correlation_score']:.4f} | "
                  f"Rotation: {result['optimal_rotation']:3d} pixels | "
                  f"GPS: ({result['gps']['northing']:.2f}, {result['gps']['easting']:.2f})")
            print(f"      Match: {os.path.basename(result['image_path'])}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    total_queries = len(all_results)
    successful_queries = len([r for r in all_results.values() if r])
    
    print(f"Total queries processed: {total_queries}")
    print(f"Successful queries: {successful_queries}")
    print(f"Failed queries: {total_queries - successful_queries}")
    
    if successful_queries > 0:
        all_scores = [result['correlation_score'] 
                     for results in all_results.values() 
                     for result in results if results]
        
        if all_scores:
            print(f"Best overall score: {max(all_scores):.4f}")
            print(f"Average best score: {np.mean([max([r['correlation_score'] for r in results]) for results in all_results.values() if results]):.4f}")
            print(f"Worst best score: {min([max([r['correlation_score'] for r in results]) for results in all_results.values() if results]):.4f}")

def save_results_to_file(all_results, query_dir, output_file):
    """Save all query results to a JSON file."""
    output_data = {
        'query_directory': query_dir,
        'timestamp': pd.Timestamp.now().isoformat(),
        'total_queries': len(all_results),
        'results': {}
    }
    
    for query_filename, results in all_results.items():
        query_results = []
        for result in results:
            # Convert numpy arrays to lists for JSON serialization
            result_copy = result.copy()
            result_copy['skyline_vector'] = result_copy['skyline_vector'].tolist()
            query_results.append(result_copy)
        
        output_data['results'][query_filename] = query_results
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Query the skyline database with a folder of images.")
    parser.add_argument("--query_dir", required=True, help="Path to the directory containing query images.")
    parser.add_argument("--database", required=True, help="Path to the skyline database JSON file.")
    parser.add_argument("--sam_checkpoint", default="sam_vit_b_01ec64.pth", help="Path to the SAM model checkpoint.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top matches to return per query.")
    parser.add_argument("--save_results", help="Optional path to save results as JSON file.")
    parser.add_argument("--show_details", action="store_true", help="Show detailed comparison information.")
    parser.add_argument("--image_extensions", nargs='+', default=['.png', '.jpg', '.jpeg', '.bmp', '.tiff'], 
                       help="Image file extensions to process (default: .png .jpg .jpeg .bmp .tiff)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.query_dir):
        print(f"Error: Query directory not found at {args.query_dir}")
        return
    
    if not os.path.isdir(args.query_dir):
        print(f"Error: {args.query_dir} is not a directory")
        return
    
    if not os.path.exists(args.database):
        print(f"Error: Database file not found at {args.database}")
        return
    
    if not os.path.exists(args.sam_checkpoint):
        print(f"Error: SAM checkpoint not found at {args.sam_checkpoint}")
        return
    
    # Find all query images
    query_images = []
    for filename in os.listdir(args.query_dir):
        if any(filename.lower().endswith(ext.lower()) for ext in args.image_extensions):
            query_images.append(os.path.join(args.query_dir, filename))
    
    if not query_images:
        print(f"Error: No images found in {args.query_dir} with extensions {args.image_extensions}")
        return
    
    print(f"Found {len(query_images)} query images in {args.query_dir}")
    
    # Setup
    print("Setting up SAM predictor...")
    predictor = setup_sam_predictor(checkpoint_path=args.sam_checkpoint)
    
    print("Loading skyline database...")
    database = load_database(args.database)
    print(f"Loaded database with {len(database)} entries.")
    
    # Query the database with all images
    all_results = query_skyline_database(query_images, database, predictor, args.top_k)
    
    # Display results
    print_results(all_results)
    
    # Show detailed information if requested
    if args.show_details and all_results:
        print(f"\n{'='*80}")
        print("DETAILED ANALYSIS")
        print(f"{'='*80}")
        
        # Show details for the query with the best overall match
        best_query = None
        best_score = -1
        
        for query_filename, results in all_results.items():
            if results and results[0]['correlation_score'] > best_score:
                best_score = results[0]['correlation_score']
                best_query = query_filename
        
        if best_query:
            print(f"Best performing query: {best_query}")
            query_path = os.path.join(args.query_dir, best_query)
            query_skyline = extract_skyline_from_image(query_path, predictor)
            
            if query_skyline is not None:
                print(f"Query skyline statistics:")
                print(f"  Length: {len(query_skyline)} pixels")
                print(f"  Min height: {np.min(query_skyline)}")
                print(f"  Max height: {np.max(query_skyline)}")
                print(f"  Mean height: {np.mean(query_skyline):.1f}")
                print(f"  Std deviation: {np.std(query_skyline):.1f}")
                
                best_match = all_results[best_query][0]
                print(f"\nBest match analysis:")
                print(f"  Database skyline length: {len(best_match['skyline_vector'])}")
                print(f"  Correlation score: {best_match['correlation_score']:.6f}")
                print(f"  Optimal rotation: {best_match['optimal_rotation']} pixels")
                print(f"  GPS coordinates: ({best_match['gps']['northing']:.6f}, {best_match['gps']['easting']:.6f})")
    
    # Save results if requested
    if args.save_results:
        import pandas as pd  # Import here since it's optional
        save_results_to_file(all_results, args.query_dir, args.save_results)

if __name__ == "__main__":
    main()