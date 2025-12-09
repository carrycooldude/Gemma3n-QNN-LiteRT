package com.example.qnn_litertlm_gemma

import android.content.Context
import android.util.Log
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.LogSeverity
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.SamplerConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Singleton manager for LiteRT-LM Engine
 * Handles model initialization and conversation management
 */
class LiteRTLMManager private constructor(private val context: Context) {
    
    private var engine: Engine? = null
    private var conversation: com.google.ai.edge.litertlm.Conversation? = null
    private var isInitialized = false
    
    companion object {
        private const val TAG = "LiteRTLMManager"
        private const val MODEL_FILENAME = "gemma3n.litertlm"
        
        @Volatile
        private var INSTANCE: LiteRTLMManager? = null
        
        fun getInstance(context: Context): LiteRTLMManager {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: LiteRTLMManager(context.applicationContext).also { INSTANCE = it }
            }
        }
    }
    
    /**
     * Initialize the LiteRT-LM engine with the model
     * This should be called on a background thread as it can take several seconds
     */
    suspend fun initialize(modelPath: String): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            if (isInitialized) {
                Log.d(TAG, "Engine already initialized")
                return@withContext Result.success(Unit)
            }
            
            Log.d(TAG, "Initializing LiteRT-LM engine...")
            
            // Try GPU first, fallback to CPU
            val backend = detectBestBackend()
            Log.d(TAG, "Using backend: $backend")
            
            val engineConfig = EngineConfig(
                modelPath = modelPath,
                backend = backend,
                cacheDir = context.cacheDir.path,
                visionBackend = backend,  // Gemma3n supports multimodal
                audioBackend = Backend.CPU
            )
            
            engine = Engine(engineConfig)
            engine?.initialize()
            
            // Create a conversation with default configuration
            val conversationConfig = ConversationConfig(
                systemMessage = Message.of("You are Gemma, a helpful AI assistant powered by Google's LiteRT-LM running on device."),
                samplerConfig = SamplerConfig(
                    topK = 40,
                    topP = 0.95,
                    temperature = 0.8
                )
            )
            
            conversation = engine?.createConversation(conversationConfig)
            isInitialized = true
            
            Log.d(TAG, "LiteRT-LM initialized successfully")
            Result.success(Unit)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize LiteRT-LM", e)
            Result.failure(e)
        }
    }
    
    /**
     * Send a message and receive streaming response
     */
    suspend fun sendMessage(messageText: String): Flow<Message> {
        if (!isInitialized || conversation == null) {
            throw IllegalStateException("LiteRT-LM not initialized. Call initialize() first.")
        }
        
        return withContext(Dispatchers.IO) {
            val userMessage = Message.of(messageText)
            conversation!!.sendMessageAsync(userMessage)
        }
    }
    
    /**
     * Detect the best available backend
     */
    private fun detectBestBackend(): Backend {
        // Try GPU first (which includes QNN/NPU support via delegates if applicable, but explicit NPU backend failed for this model)
        return try {
            Log.d(TAG, "Attempting to use GPU backend...")
            Backend.GPU
        } catch (e: Exception) {
            Log.w(TAG, "GPU backend not available, falling back to CPU", e)
            Backend.CPU
        }
    }
    
    /**
     * Get the model file path
     */
    fun getModelPath(): String {
        return File(context.filesDir, MODEL_FILENAME).absolutePath
    }
    
    /**
     * Check if model file exists
     */
    fun isModelDownloaded(): Boolean {
        return File(getModelPath()).exists()
    }
    
    /**
     * Clean up resources
     */
    fun cleanup() {
        try {
            conversation?.close()
            engine?.close()
            conversation = null
            engine = null
            isInitialized = false
            Log.d(TAG, "LiteRT-LM resources cleaned up")
        } catch (e: Exception) {
            Log.e(TAG, "Error cleaning up resources", e)
        }
    }
}
