import 'dart:async';
import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import '../models/research_paper.dart';

class PaperService {
  /// Loads research papers from the assets/papers.json file
  /// Returns a List<ResearchPaper> parsed from the JSON data
  static Future<List<ResearchPaper>> loadPapers() async {
    try {
      // Load the JSON string from assets
      final String jsonString = await rootBundle.loadString(
        'assets/papers.json',
      );

      // Decode the JSON string into a dynamic object
      final dynamic jsonData = jsonDecode(jsonString);

      final List<ResearchPaper> papers = [];

      // If JSON is a plain list (legacy format), parse each entry
      if (jsonData is List) {
        for (final item in jsonData) {
          if (item is Map<String, dynamic>) {
            papers.add(ResearchPaper.fromJson(item));
          }
        }
      }
      // If JSON is a map of categories -> list of papers, iterate categories
      else if (jsonData is Map<String, dynamic>) {
        jsonData.forEach((category, value) {
          if (value is List) {
            for (final item in value) {
              if (item is Map<String, dynamic>) {
                // Build ResearchPaper from the legacy fields and attach category
                papers.add(
                  ResearchPaper(
                    id: item['id'] as int?,
                    title: item['title'] as String? ?? '',
                    link: item['link'] as String? ?? '',
                    abstract: item['summary'] as String? ?? '',
                    introduction: '',
                    materialsMethods: '',
                    results: '',
                    discussion: '',
                    simplifiedAiVersion: item['summary'] as String? ?? '',
                    keywords: item['keywords'] != null
                        ? List<String>.from(item['keywords'] as List)
                        : [],
                    category: category,
                  ),
                );
              }
            }
          }
        });
      } else {
        throw Exception('Unsupported JSON structure for papers');
      }

      return papers;
    } catch (e) {
      throw Exception('Failed to load papers: $e');
    }
  }

  /// Fetches paper data from the API with detailed sections
  /// Returns a ResearchPaper with abstract, introduction, materials_methods, results, discussion, and simplified_ai_version
  static Future<ResearchPaper> fetchPaperFromApi(String paperUrl) async {
    try {
      // Construct the API URL with proper query parameter (use HTTP as requested)
      final apiUrl = Uri.parse(
        'https://astrolens.mohamedayman.net/summarize-get',
      ).replace(queryParameters: {'url': paperUrl});

      print('DEBUG: Calling API with URL: $apiUrl');

      // Make the HTTP request with headers and extended timeout
      final client = http.Client();
      try {
        final response = await client
            .get(
              apiUrl,
              headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'AstroLens-Flutter/1.0',
              },
            )
            .timeout(
              const Duration(
                seconds: 60,
              ), // Allow up to 60 seconds for API response
            );

        print('DEBUG: API Response Status: ${response.statusCode}');
        // print('DEBUG: API Response Body: ${response.body}');

        if (response.statusCode == 200) {
          // Parse the JSON response
          final Map<String, dynamic> jsonData = jsonDecode(response.body);

          // Create ResearchPaper from API response
          return ResearchPaper.fromApiJson(jsonData);
        } else {
          throw Exception(
            'Failed to fetch paper data: ${response.statusCode} - ${response.body}',
          );
        }
      } finally {
        client.close();
      }
    } catch (e) {
      print('DEBUG: Error in fetchPaperFromApi: $e');
      throw Exception('Error fetching paper from API: $e');
    }
  }

  /// Alternative static function if you prefer not to use a class
  /// This function can be called directly without instantiating the class
  static Future<List<ResearchPaper>> getPapersFromAssets() async {
    return await loadPapers();
  }
}

/// Standalone function for loading papers (alternative approach)
/// This can be used if you prefer a simple function over a class method
Future<List<ResearchPaper>> loadPapers() async {
  // Delegate to PaperService.loadPapers to handle both list and categorized map formats.
  return await PaperService.loadPapers();
}
