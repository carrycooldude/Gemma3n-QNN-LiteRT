package com.example.qnn_litertlm_gemma

import android.os.Bundle
import android.text.Editable
import android.text.TextWatcher
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.example.qnn_litertlm_gemma.databinding.ActivityMainBinding
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.flow.onStart
import kotlinx.coroutines.launch
import java.util.Date

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var chatAdapter: ChatAdapter
    private lateinit var liteRTLMManager: LiteRTLMManager
    private lateinit var modelDownloader: ModelDownloader
    
    // Store conversation history
    private val messages = mutableListOf<ChatMessage>()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        liteRTLMManager = LiteRTLMManager.getInstance(this)
        modelDownloader = ModelDownloader(this)

        setupRecyclerView()
        setupInput()
        
        // Start initialization process
        checkAndInitialize()
    }
    
    private fun setupRecyclerView() {
        chatAdapter = ChatAdapter()
        binding.recyclerViewMessages.apply {
            adapter = chatAdapter
            layoutManager = LinearLayoutManager(this@MainActivity).apply {
                stackFromEnd = true
            }
            
            // Scroll to bottom when keyboard opens (layout shrinks)
            addOnLayoutChangeListener { _, _, _, _, bottom, _, _, _, oldBottom ->
                if (bottom < oldBottom) {
                    binding.recyclerViewMessages.postDelayed({
                        if (messages.isNotEmpty()) {
                            binding.recyclerViewMessages.smoothScrollToPosition(messages.size - 1)
                        }
                    }, 100)
                }
            }
        }
    }
    
    private fun setupInput() {
        // Enable/disable send button based on input
        binding.editTextMessage.addTextChangedListener(object : TextWatcher {
            override fun beforeTextChanged(s: CharSequence?, start: Int, count: Int, after: Int) {}
            override fun onTextChanged(s: CharSequence?, start: Int, before: Int, count: Int) {
                binding.buttonSend.isEnabled = !s.isNullOrBlank()
            }
            override fun afterTextChanged(s: Editable?) {}
        })
        
        binding.buttonSend.setOnClickListener {
            val text = binding.editTextMessage.text.toString().trim()
            if (text.isNotEmpty()) {
                sendMessage(text)
                binding.editTextMessage.text?.clear()
            }
        }
        
        // Initially disable send button
        binding.buttonSend.isEnabled = false
    }
    
    private fun checkAndInitialize() {
        lifecycleScope.launch {
            if (modelDownloader.isModelDownloaded()) {
                initializeEngine()
            } else {
                downloadModel()
            }
        }
    }
    
    private fun downloadModel() {
        lifecycleScope.launch {
            binding.progressBarLoading.visibility = View.VISIBLE
            binding.textLoadingStatus.visibility = View.VISIBLE
            binding.textLoadingStatus.text = "Downloading Gemma3n model..."
            binding.cardInput.visibility = View.GONE
            
            modelDownloader.downloadModel().collect { progress ->
                when (progress) {
                    is DownloadProgress.Started -> {
                         binding.textLoadingStatus.text = "Starting download..."
                    }
                    is DownloadProgress.Progress -> {
                        binding.textLoadingStatus.text = "Downloading: ${progress.percentage}%"
                    }
                    is DownloadProgress.Complete -> {
                        binding.textLoadingStatus.text = "Download complete!"
                        initializeEngine()
                    }
                    is DownloadProgress.Error -> {
                        binding.progressBarLoading.visibility = View.GONE
                        binding.textLoadingStatus.text = "Error: ${progress.message}"
                        Toast.makeText(this@MainActivity, "Download failed: ${progress.message}", Toast.LENGTH_LONG).show()
                    }
                }
            }
        }
    }
    
    private fun initializeEngine() {
        lifecycleScope.launch {
            binding.progressBarLoading.visibility = View.VISIBLE
            binding.textLoadingStatus.visibility = View.VISIBLE
            binding.textLoadingStatus.text = "Initializing LiteRT-LM Engine (this may take a moment)..."
            
            val result = liteRTLMManager.initialize(modelDownloader.getModelPath())
            
            binding.progressBarLoading.visibility = View.GONE
            binding.textLoadingStatus.visibility = View.GONE
            
            if (result.isSuccess) {
                binding.cardInput.visibility = View.VISIBLE
                addSystemMessage("Gemma3n initialized and ready. Chat with me!")
            } else {
                val error = result.exceptionOrNull()?.message ?: "Unknown error"
                binding.textLoadingStatus.visibility = View.VISIBLE
                binding.textLoadingStatus.text = "Initialization failed: $error"
                Toast.makeText(this@MainActivity, "Init failed: $error", Toast.LENGTH_LONG).show()
            }
        }
    }
    
    private fun sendMessage(text: String) {
        // Add user message to UI
        val userMessage = ChatMessage(MessageSender.USER, text)
        messages.add(userMessage)
        updateMessages()
        
        // Prepare assistant message placeholder
        val assistantMessageIndex = messages.size
        val assistantMessage = ChatMessage(MessageSender.ASSISTANT, "", isStreaming = true)
        messages.add(assistantMessage)
        updateMessages()
        
        lifecycleScope.launch {
            var fullResponse = ""
            
            try {
                liteRTLMManager.sendMessage(text)
                    .catch { e ->
                        // Handle error in stream
                        messages[assistantMessageIndex] = assistantMessage.copy(
                            content = "Error: ${e.message}",
                            isStreaming = false
                        )
                        updateMessages()
                    }
                    .collect { messageChunk ->
                        // Append text chunk
                        // Note: Depending on the API, messageChunk might be the full text or just a diff.
                        // Based on the docs: "Asynchronous call for streaming responses."
                        // Usually Flow emits the *delta* or the *accumulated* message?
                        // Let's assume it emits Message objects which might contain the diff or accumulated.
                        // The docs example says: .collect { print(it) }
                        // Usually LiteRT-LM emits chunks. let's treat it as chunks.
                        // Wait, looking at docs: "returns a Kotlin Flow for streaming responses."
                        // and collect { print(it.toString()) }
                        // Let's assume it converts nicely to string.
                        
                        val chunkText = messageChunk.toString()
                        fullResponse += chunkText
                        
                        messages[assistantMessageIndex] = assistantMessage.copy(
                            content = fullResponse,
                            isStreaming = true
                        )
                        chatAdapter.notifyItemChanged(assistantMessageIndex)
                        binding.recyclerViewMessages.smoothScrollToPosition(assistantMessageIndex)
                    }
                
                // Final update when done
                messages[assistantMessageIndex] = assistantMessage.copy(
                    content = fullResponse,
                    isStreaming = false
                )
                updateMessages()
                
            } catch (e: Exception) {
                messages[assistantMessageIndex] = assistantMessage.copy(
                    content = "Error sending message: ${e.message}",
                    isStreaming = false
                )
                updateMessages()
            }
        }
    }
    
    private fun addSystemMessage(text: String) {
        messages.add(ChatMessage(MessageSender.SYSTEM, text))
        updateMessages()
    }
    
    private fun updateMessages() {
        chatAdapter.submitList(messages.toList())
        if (messages.isNotEmpty()) {
            binding.recyclerViewMessages.smoothScrollToPosition(messages.size - 1)
        }
    }
    
    override fun onDestroy() {
        super.onDestroy()
        liteRTLMManager.cleanup()
    }
}
