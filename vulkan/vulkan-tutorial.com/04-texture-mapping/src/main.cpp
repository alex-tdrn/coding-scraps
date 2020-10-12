#include "TextureMapping.h"

#include <iostream>

int main()
{
	TextureMapping app;

	try
	{
		app.run();
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}
